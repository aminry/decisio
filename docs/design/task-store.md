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

- **Monitors** (the answer-mix counts of a certified task) stay per replica and are not shared. Their metrics (`decisio_task_*` in `/metrics`) are per replica and are summed in Prometheus. No fleet monitoring is built for them; the docs say so (a fast table that learns from traffic has a cluster view: "Learning on the shared store" below).
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

## Learning on the shared store

Status: built (decisio.serve.fast_table_replicas).
A fast table that learns from traffic (`"fast_table": {"learn": true}`, decisio.serve.fast_table_learn) learns across the replicas that share a store.
One server's learning, with `--tasks-dir` or in memory, is unchanged; this section is only about where the rows, the versions and the state live when replicas share them.

### Layout

Under the store's prefix, beside `MANIFEST.json` and `objects/`:

```text
learner/LEASE.json                                  {"holder", "generation", "expires"}: the one replica that trains
tasks/<digest>/STATE.json                           the task's learned-version state, moved only by compare-and-swap
tasks/<digest>/versions/<sha256>.json               one promoted version, immutable, named by the digest of its bytes
tasks/<digest>/log/<replica>/<start ms>-<seq>.seg   one replica's rows, written once (create-only)
tasks/<digest>/log/merged/<gen>-<token>-<n>.seg     rows the trainer compacted, written once
tasks/<digest>/log/INDEX.json                       the merged files and the replica segments they hold
tasks/<digest>/labels.npz                           a labelled task's examples' features, written at registration
tasks/<digest>/stats/<replica>.json                 one replica's counts and guards, every 5 minutes
```

`<digest>` names one registration of a task id: a registration on a shared store carries a random `learn.registration`, and a task registered again starts afresh under a new digest, as one server drops its rows.
The replica that removes or registers a task again deletes the old registration's objects; a segment a replica writes before its next poll is left behind, and `python -m decisio.serve.task_store_shared gc` removes it with the unreferenced task objects.
A replica's id is its host name and a random suffix, made at start: a restarted replica is a new one.

### Logs

Every replica that writes to the store logs the rows decisio reads, as one server does, and each row carries the replica's id and serving-stack fingerprint.
Rows are buffered in memory and written as a new segment at 4 MB or 5 minutes, whichever comes first, and only when there are rows.
The request path never calls the store: the task store's poll thread writes the segments.
A segment's name and bytes are fixed at its first write, so a write that reached the store but whose answer was lost is found there on the retry, and its rows are counted once.
The trainer reads every replica's segments at each refit, keeps the newest `max_rows`, and compacts: once 16 segments are not yet merged, or rows fall past `max_rows`, it writes the kept rows as merged files (at most 64 MB each), names them and the segments they hold in `INDEX.json` by compare-and-swap, and only then deletes what they replace.
A file that cannot be read is named in a warning and left in place, never compacted away.

### One trainer: the lease and the fence

The trainer is the holder of `learner/LEASE.json`.
A replica takes it with a create when there is none, or with a compare-and-swap on the copy it read when that copy has expired (or is its own, after a failed renewal); each take is generation + 1.
The lease lasts 120 seconds and the holder renews it every 40 seconds by a compare-and-swap on its own last write.
A renewal that fails, because another replica wrote the lease or the store cannot be reached, stops training at once.
A replica tries for the lease only while it has a learning task, and releases it at a clean stop, so a successor takes it at its next try.
Expiry is judged on each replica's clock, so a holder whose clock runs behind can believe it holds a lease another replica has taken; the fence below is what makes that harmless.

Every published version carries its lease generation.
A publish writes the version's object, reads the lease again, and refuses if the generation is no longer its own; the state's compare-and-swap then refuses if a later generation has published.
A holder that lost the lease while it was fitting (a pause, a partition, a slow clock) publishes nothing, and its refit's outcome is `fenced`.
The trainer is the lease holder's existing refit worker, at the same low priority; a separate learner process that holds the lease and serves nothing is not built (below).

### Publishing

Versions are content-addressed immutable objects; what moves is the task's state object, `tasks/<digest>/STATE.json`: the kept versions (number, object, generation), the live one, an epoch that moves at every promotion or rollback, a suspension, the salt of the rows' held-out split, the last refit's outcome and the cluster view.
It is written only by compare-and-swap on its ETag, and a write that loses to another replica's reads it again and retries.

The state is its own object per task rather than an entry kind in `MANIFEST.json`, for three reasons.
A replica on 0.41.0 reads every manifest entry as a task of a kind it knows: an entry of a new kind fails its start, and every later poll fails at that entry, so a new entry kind would break a rolling upgrade; it never reads `tasks/`.
Learning state changes on its own schedule (a suspension, a refit's outcome, the cluster view every 5 minutes), and in its own object it never contends with task registrations, nor makes every replica fetch the whole manifest again.
The cost is one more conditional GET per learning task per poll, an empty 304 when nothing changed.

Every replica follows the state at the task store's poll (`--task-store-poll-seconds`).
On a change it fetches the versions it does not hold (each once: they are immutable), checks each against its digest and that its feature function (`fast_table.record_features`) is one of the task's students, and swaps the live one in one assignment; a state that does not check out is not applied at all, and the replica keeps what it had.
A version is then timed on the replica's own requests before its table answers there, as a labelled table is on a new stack (`fast_table.SHADOW_EVERY`, `SHADOW_N`, `timing_guard`): the host differs, and decisio answers meanwhile, its rows logged as `no_table`.
A version whose feature function this server cannot compute, or that was certified on another serving stack, is `off` on that replica, as on one server.

A replica's alarm (either drift guard) suspends the table on that replica at once and, at its next poll, writes the suspension to the state, which every replica follows: the table is suspended for all, as `suspend_table` does for a labelled table.
A suspension names the epoch it was raised under, so it never suspends a version that went live after it.
A rollback (`POST /v1/tasks/{id}/fast_table/rollback`) on any replica that writes moves the state the same way.

### The certificate across replicas

The trainer fits and certifies exactly as one server does, on the union of every replica's rows.
Only rows whose serving-stack fingerprint is the trainer's are fitted on or certified on, and the certificate is made only on held-out `no_table` and `explored` rows: spot checks come only from the table's own answers, so they feed the disagreement CUSUM and never a certificate, on any replica.
The rows' held-out split is keyed by a salt in the state, so a request's row falls on the same side whichever replica logged it.
The certificate records the replicas it drew on and each one's held-out rows (`replicas`), and the held-out rows it left out because another stack logged them (`other_stack_rows`).
A fleet that wants every replica's traffic in the certificate runs one stack.

### Restart

A replica that starts loads the task records, reads each learning task's state, fetches its versions, and serves the live one once its own timing passes; it keeps logging, and nothing is learned again.
A whole cluster that restarts resumes the same way; the first replica to try after the old lease is released (or has expired) takes it, with the next generation.

### Read-only replicas

A replica started with `--task-store-readonly` never writes to the store: it does not take the lease, does not log rows, does not write its stats, and does not share an alarm (its own suspends it alone); a rollback there answers 409.
It follows the state and serves the versions, after its own timing, as any replica does.
It does not log because the flag already means "this replica's credentials may be read-only", and a replica that tried to write with them would only fill its buffer and drop rows.
The certificate is then made on the writing replicas' rows: with a load balancer that spreads requests evenly, those are a random share of the same traffic.

### Monitoring

Every replica that writes puts `stats/<replica>.json` every 5 minutes (its rows, its state, its guards, its observed coverage).
The trainer reads them on the same schedule and writes the cluster view into the state, so `GET /v1/tasks/{id}` on any replica shows it under `status.fast_table.cluster`, as of the trainer's last read; the replica's own view is beside it (`status.fast_table.replica`, `rows.this_replica`).
A replica silent for 15 minutes has stopped: its stats file is deleted and its rows stay in the count.
The trainer's refit trigger (2,000 rows or an hour) counts every replica's rows from these stats, so it can fire up to 5 minutes late.
`/health` says, under `systemone.fast_table_learning.shared`, this replica's id, the lease, the buffer and the last poll.

### When the store cannot be reached

Serving never waits on the store: a replica serves its cached versions, its guards keep running, and its rows wait in memory, up to 256 MB for the server (all tasks together).
Past that, new rows are dropped and counted (`/health`: `shared.buffer.rows_dropped`).
The lease lapses, since it cannot be renewed, so nobody promotes; the holder stops training at its first failed renewal.
When the store comes back, the waiting segments are written oldest first, and the next replica to try takes the lease.

### Request costs

For one learning task, per replica:

- each poll (every 5 seconds by default): one conditional GET of the state, besides the manifest's;
- every 40 seconds: one lease request (the holder's renewal is a conditional PUT, the others' a GET);
- at most every 5 minutes, or every 4 MB of rows: one PUT of a segment;
- every 5 minutes: one PUT of its stats;
- when a version is promoted: one GET of the version (tens of KB).

On the trainer, also:

- every 5 minutes: a LIST of the stats, one GET per replica, and one conditional PUT of the state;
- at each refit (every 2,000 rows or every hour): a LIST of the log, a GET of its index and of every merged and unmerged file, that is the task's kept rows (about 0.6 KB a row with the static student, so about 120 MB at the default 200,000), and one or two state PUTs; at a promotion one PUT of the version and one GET of the lease;
- at a compaction: one PUT per merged file, one PUT of the index, and one DELETE per file replaced.

With the default poll that is about 20,000 requests a day per replica for one learning task, on top of the manifest's 17,280: 17,280 conditional GETs of the state (almost all of them empty 304s), 2,160 lease requests and a few hundred PUTs.
The trainer also reads the task's kept rows once per refit.

### Not built

- A learner process (`python -m decisio.serve.learner`) that holds the lease and serves nothing.
  The certificate counts only the rows logged on the trainer's serving stack, and a process with no engine has no stack; giving it one to train for is a decision of its own, not a cheap addition.
- A cache of the trainer's segments between refits (each refit reads the kept rows again).
