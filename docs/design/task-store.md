<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: a shared task store for replicas

Status: accepted and built (part C of the task-persistence work; part B, `--tasks-dir`, is separate). All four recommendations were accepted in "Decisions asked for" below. What the build settled beyond the proposal is under "As built" at the end; the user documentation is `tasks.md` ("Replicas: a shared store").

## The problem

Taught tasks (calibration and head tasks, abstention tasks) belong to one server.
With `--tasks-dir` they survive that server's restart.
Replicas behind a load balancer each hold their own: a task taught on one is not on the others, and a request that lands on another replica is answered untaught.
Exporting and importing by hand (`GET /v1/tasks?full=1`, `POST /v1/tasks/import`) works for one-off changes and does not keep a fleet in step.

What a fleet needs:

1. A task registered on any replica reaches every replica, as the exact fitted record (no re-fit: a re-fit on another card could give other numbers), within seconds.
2. Two registrations cannot silently overwrite each other.
3. A delete reaches every replica.
4. A replica checks the record against its own stack, as an import does today: another model or rendering is kept and not applied, a certificate made under another serving stack is suspended.
5. No extra service to run if it can be avoided, and credentials from the usual places.

## What is already there

Part B gives the store an interface of three operations: `save(kind, record)`, `delete(kind, task_id)`, `load()`, with `DirBackend` as the first implementation and a write that fails failing the request (`decisio.serve.task_backends`).
A task record is the JSON `GET /v1/tasks?full=1` returns: self-contained, with its fingerprint and, for a certified task, its certificate and the stack it was made under.
An import (`TaskStore.load`) already applies such a record without scoring anything.
So the shared store is a second backend that adds two things the directory does not need: **versions** (to detect a concurrent write) and **change notification** (to learn what the others wrote).

## Options for the backend

| | Object storage (S3-compatible) | A database (Postgres) | A shared filesystem (NFS, EFS) with `DirBackend` | A key-value or coordination service (etcd, Consul, Redis) |
| --- | --- | --- | --- | --- |
| Conditional write | `If-Match` / `If-None-Match` on PutObject (S3, R2 and MinIO have conditional writes, to be confirmed against their current documentation and the probe below before building; GCS has it natively as a generation match, and the S3-interop API's support is to be probed, not assumed) | Row versions, transactions | None that holds across clients (rename is atomic on one host; NFS locking is advisory and weak) | Compare-and-swap, yes |
| Propagation | Polling a small manifest (one conditional GET per replica every few seconds) | `LISTEN/NOTIFY` or polling | Polling directory mtimes | Watches |
| Record size (a head is up to a few MB for 150 options) | Fine | Fine (a `bytea`/`text` column) | Fine | Over the usual limits (etcd 1.5 MB per value, Consul 512 KB) |
| What a customer runs | A bucket (they have one), credentials from the environment | A database and a driver, migrations, a connection pool | A mount | A cluster they would not otherwise run |
| New dependency in decisio | An object-storage client | A Postgres driver | None | A client |
| Cost of a small fleet | Cents a month | A managed instance | The mount | A cluster |

**Recommendation: object storage.** It is the least to operate, tasks are write-rarely and read-at-start, the record sizes fit, and conditional writes give exactly the one primitive needed.
A database is the better answer for a customer who already runs one and wants transactions across other data; it can be a second backend behind the same interface later, and nothing here depends on the choice.
A shared filesystem is not recommended: it gives no conditional write, so two replicas registering at once can lose a task without anyone noticing.

Client library, for the build: **`obstore`** (Python bindings of the Rust `object_store`: one API for S3-compatible stores, GCS and Azure, conditional put by ETag or generation, each provider's own credential chain, a small wheel) against **`boto3`** (ubiquitous, S3 only, a large dependency).
Either sits behind a five-method `ObjectStore` protocol (`get` with `if_none_match`, `put` with `if_match` or create-only, `delete`, `head`, `list`), so the choice is an optional extra (`decisio[task-store]`) and the tests use an in-memory fake of the protocol.
Before building I would check that obstore's wheels cover the platforms we ship (Linux x86_64/arm64, macOS arm64) and that its conditional put works against MinIO and R2; if not, boto3.
A start-up probe writes, conditionally rewrites and deletes a probe object and refuses to start if the store does not enforce preconditions (a store that silently ignores `If-Match` would turn the conflict rule into last-writer-wins without anyone knowing).

## The layout and the protocol

Under `s3://bucket/prefix/`:

```
objects/<sha256>.json   one immutable task record per fitted version, named by the digest of its bytes
MANIFEST.json           {"format", "version": n, "entries": {"<kind>/<id-digest>": {"id", "version", "object": sha256, "deleted": false}}}
```

- **Register** (on any replica, after the fit, the certification and the served check, exactly as today): (1) put `objects/<sha256>.json` (idempotent; a second put of the same bytes is a no-op); (2) read the manifest and its ETag; (3) if the entry for this task has the version this replica last applied (or none, and this replica holds none), write the manifest with that entry at version + 1 pointing at the new object, `If-Match: <ETag>`; (4) on a failed precondition, re-read the manifest and go to (3) if only *other* tasks changed; if *this* task changed, answer 409 (below); (5) apply the record locally and answer. The response is sent only after step 3 succeeded: a task is never reported registered that the store does not hold.
- **Delete**: the same compare-and-swap writes the entry with `"deleted": true` and version + 1 (a tombstone, so replicas that have not polled learn of it; tombstones older than a retention period, 7 days by default, are dropped by whichever replica next writes the manifest).
- **Follow**: every `--task-store-poll-seconds` (default 5) a replica issues `GET MANIFEST.json` with `If-None-Match: <its ETag>`; an unchanged manifest costs one empty 304. On a change it fetches the objects of the entries whose version is newer than the one it applied, verifies each against its sha256, and applies them like an import (`TaskStore.load`: a record of another model or rendering is kept and not applied, a certificate made under another serving stack is suspended, nothing is re-fitted); tombstones remove the task. A replica starts by doing the same with everything, and refuses to start if it cannot reach the store (an unreachable store at start must not become untaught answers).
- **The store goes away later**: the replica keeps serving what it has applied, registrations and removals answer 503, `/health` stays 200 and says `task_store: {"ok": false, "last_sync_age_s": ...}` (a replica is not killed because storage is down), and the age is a metric.
- Manifest and objects together are small (a manifest entry is under 200 bytes); a fleet with thousands of tasks still polls one object.

## The conflict rule

A registration on a shared store must not silently replace a task someone else changed.
Two candidate rules:

- **Last writer wins** (every successful manifest write is the new truth; retry on a failed precondition for the same task). Simple, and what a single server does today. On a fleet it hides the exact case the compare-and-swap exists to catch.
- **A write names the version it replaces** (recommended). The replica sends the version of the task it currently holds (none, for a new id). If the store holds a different version, the request fails with **409 Conflict** and the body names the store's version and the replica's, so the caller sees that someone taught the task since and decides: wait for the replicas to converge and send it again, or send it with `?replace=1` to replace whatever the store holds. A new id registered by two replicas at once: one succeeds, the other gets 409.

This costs the caller one retry in the rare race and removes the silent overwrite.
Different ids never conflict (a failed manifest precondition caused only by another id is retried inside the request).

## Who may register

Registration is the expensive operation: it reads the engine for every example (minutes for a certified task on a big list) and holds that replica's registration lock, and the replica's card does the fitting.
- **Any replica registers** (the compare-and-swap makes it safe). Simplest to run: the load balancer needs no routing. A replica that is fitting is busy for that time; its traffic is not blocked (reads and registrations are separate paths) but it is slower.
- **One designated replica registers** (`--task-store-readonly` on the others: `POST/DELETE` on `/v1/tasks*` and `/v1/abstention/tasks*` answer 409 naming the designated one). The fitting load is isolated, and every record is fitted on one card, so the fleet's numbers come from one place. The load balancer routes the task-writing paths to it.

**Recommendation: allow any replica, and offer the read-only flag.** The compare-and-swap is the safety net either way, the flag makes a designated teacher a one-line setting, and the default does not make a fleet depend on one replica being up to teach.
Mixed cards: a certificate carries the stack it was made under, so on a replica with another card or engine version the task is served uncertified (`serving_stack_changed`) until it is certified there. A fleet that wants certified answers everywhere runs one stack; the documentation says so.

## What does not change

- **Monitors** (the answer-mix counts of a certified task) stay per replica and are not shared. Their metrics (`decisio_task_*` in `/metrics`) are per replica and are summed in Prometheus. No fleet monitoring is built; the docs say so.
- **Reads are eventually consistent.** After a registration, other replicas serve the new version within the poll interval plus the fetch (seconds), and until then the old one; a client that must read its own write talks to the replica that took it. The docs say so, with the interval.
- **Credentials** come from the SDK chain or the environment (instance role, workload identity, `AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`), never from a flag or a file decisio writes, and are not logged; the endpoint URL is logged without a query string or userinfo.
- `--tasks-dir` and the shared store are alternatives, not a stack: a replica uses one of them. (A local cache of the shared store in a directory, so that a restart works while the store is down, is a later option, not part of this.)
- A replica does not apply a record it cannot verify: a digest that does not match, a record of another format, or a version that goes backwards is refused and logged, and the replica keeps what it had.

## Flags and documentation

`--task-store s3://bucket/prefix` (also `gs://` and `az://` if obstore is chosen), `--task-store-endpoint URL` (MinIO, R2, GCS interop), `--task-store-poll-seconds N` (5), `--task-store-readonly`, `--require-tasks N` as in part B. `GET /v1/tasks` and `/health` report the store (`saved_to`, the manifest version applied, the last sync age).
Docs: `tasks.md` (the shared store), `operating.md` (the replicas section: what is and is not shared, the poll interval, mixed cards), flags and API reference.

## Tests (no GPU, no network)

- The protocol is tested against an in-memory `ObjectStore` that enforces `If-Match` and create-only exactly as the real stores document, so the compare-and-swap, the 409, the tombstones and the poll are tested deterministically.
- A two-replica test on the CPU stand-in: two servers over one fake store; a task registered on A is answered by B within one poll, identical record bytes, B's certificate suspended when B's stack differs; concurrent registrations of the same id (one 409), of different ids (both succeed); a delete reaches B; the store failing mid-registration fails the request and registers nothing; a store that ignores preconditions is refused at start.
- An optional integration test against a local MinIO (skipped unless `DECISIO_TEST_MINIO` names one), run by hand before a release.

## Decisions asked for

1. **Backend**: object storage behind an `ObjectStore` protocol (recommended), with `obstore` as the client unless its wheels or conditional puts fail the checks above, then `boto3`; a database backend later if a customer asks.
2. **Propagation**: polling a manifest with a conditional GET, every 5 seconds by default (recommended); no push channel, no extra service.
3. **Conflict rule**: a write names the version it replaces; 409 otherwise, `?replace=1` to override (recommended), against last-writer-wins.
4. **Who registers**: any replica, with `--task-store-readonly` available for a designated teacher (recommended), against making one replica the only writer.

Also to confirm: tombstone retention (7 days), that an unreachable store at start is a refusal, and that mixed-card fleets lose certificates on the replicas whose stack differs (stated, not worked around).

## As built

- **Client:** `obstore` 0.12 (abi3 wheels for Linux x86_64 and aarch64 and macOS arm64, Python 3.11 and later), an optional extra `decisio[task-store]`, installed in the image. Its in-memory store enforces create-only writes, ETag compare-and-swap and conditional GETs, so the tests exercise the real semantics; `boto3` was not needed. A local filesystem store (`file://`) does not implement conditional writes and is refused by the start-up probe.
- **Checked on** a running RustFS (the two-replica test, `DECISIO_TEST_S3_ENDPOINT`) and SeaweedFS (the probe). MinIO's images could not be pulled (Docker Hub's `minio/minio` is gone, quay.io refused), and AWS S3, R2, GCS and Azure were not available; the probe is the user's check.
- **The version a write names** is, by default, the version the replica serves; a caller may name the one it was shown (`x-decisio-replaces`), or replace whatever the store holds (`?replace=1`). The check runs before the fit (a cheap manifest read) and again inside the compare-and-swap, so a race during a long fit is caught. A removal obeys the same rule; a task removed before a replica ever saw it is not a conflict.
- **A replica's start** loads everything the manifest names, and seeds the store from `--tasks-file` / `--abstention-tasks` only for ids the store lacks; a seed lost to another replica, or refused by a read-only replica, is reported (`file_tasks_not_saved`) and is not an error.
- **Garbage:** superseded objects are not deleted automatically; `python -m decisio.serve.task_store_shared gc` removes unreferenced ones older than a day.
- **Not built:** a local cache of the shared store (so a restart works while the store is down), a database backend, push notification, fleet monitoring.
