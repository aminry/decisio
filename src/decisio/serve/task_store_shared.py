# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A shared task store for replicas, on object storage (`--task-store s3://bucket/prefix`; docs/design/task-store.md).

Under the store's prefix:

    objects/<sha256>.json   one immutable task record per fitted version, named by the digest of its bytes
    MANIFEST.json           {"format", "version", "entries": {"<kind>/<id digest>": {"id", "version", "object",
                            "deleted", "updated"}}}: which object is each task's current record

A task is taught on any replica (unless it is `readonly`), fitted there exactly as one server fits it, and published as
the exact fitted record: the object is written (idempotent), then the manifest is rewritten with a compare-and-swap on
its ETag (create-only for the first). Every other replica polls the manifest with a conditional GET (an unchanged
manifest is an empty 304), fetches the objects whose entries are newer than the version it applied, checks each against
its digest, and applies it as an import does: nothing is re-fitted, a record of another model or rendering is kept and
not applied, a certificate made under another serving stack is suspended. A removal is a tombstone entry, kept for
`tombstone_days` so that replicas that have not polled learn of it.

A write names the version it replaces: the version this replica applied (or the one the caller was shown, header
`x-decisio-replaces`). If the store holds another, the write raises `TaskConflict` (409) and nothing is changed, unless
the caller said `replace`. A write that fails on a precondition because *another* task changed is retried.

The client is `obstore` (S3-compatible stores, GCS and Azure; credentials from the provider's usual chain or the
environment, never from decisio's flags or logs).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from decisio.serve.task_backends import (
    KINDS,
    WRITE_INTENT,
    TaskConflict,
    TaskPersistError,
    TaskStoreReadOnly,
    TaskStoreUnavailable,
    dumps,
)

log = logging.getLogger(__name__)

MANIFEST = "MANIFEST.json"
MANIFEST_FORMAT = "decisio-task-manifest/1"
TOMBSTONE_DAYS = 7
MAX_RACES = 8  # a manifest write retried because another task changed, at most this often per request


def _obstore():
    try:
        import obstore
    except ImportError as e:  # an optional dependency
        raise TaskPersistError(
            "--task-store needs the obstore package: pip install 'decisio[task-store]' (or pip install obstore)"
        ) from e
    return obstore


def sanitized(url: str) -> str:
    """The store's URL without credentials or query, for logs and /health."""
    p = urlsplit(url)
    host = p.hostname or ""
    if p.port:
        host += f":{p.port}"
    return f"{p.scheme}://{host}{p.path}" if p.scheme else url.split("?", 1)[0]


def open_store(url: str, endpoint: str | None = None):
    """The obstore store for `url` (s3://bucket/prefix, gs://, az://, memory://, file:///dir). `endpoint`: an
    S3-compatible service other than AWS (MinIO, R2, GCS interoperability); an http endpoint is allowed (local ones)."""
    _obstore()
    from obstore.store import from_url

    kw = {}
    if endpoint:
        kw["config"] = {"endpoint": endpoint}
        if endpoint.startswith("http://"):
            kw["client_options"] = {"allow_http": True}
    try:
        return from_url(url, **kw)
    except Exception as e:
        raise TaskPersistError(f"--task-store {sanitized(url)}: cannot open the store ({type(e).__name__}: {e})") from e


def entry_key(kind: str, task_id: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown task kind {kind!r}")
    return f"{kind}/{hashlib.sha256(str(task_id).encode()).hexdigest()[:32]}"


class _Raced(Exception):
    """The manifest changed between its read and its write."""


class ObjectBackend:
    """The shared task store on object storage; see the module's account. Implements the backend interface of
    decisio.serve.task_backends (`save`, `delete`, `load`, `describe`, `precheck`, `version_of`) and `sync_once`."""

    def __init__(
        self,
        store,
        label: str = "memory://",
        readonly: bool = False,
        poll_seconds: float = 5.0,
        tombstone_days: float = TOMBSTONE_DAYS,
        clock=time.time,
    ):
        self.store, self.label = store, label
        self.readonly, self.poll_seconds, self.tombstone_days = readonly, float(poll_seconds), float(tombstone_days)
        self.clock = clock
        self.applier = None  # set by SystemOne: apply(kind, records) and drop(kind, task_id)
        self.applied: dict[str, int] = {}  # entry key -> the version this replica applied (or wrote)
        self.ids: dict[str, tuple[str, str]] = {}  # entry key -> (kind, task id), to drop a task whose entry is gone
        self._tag = None  # (e_tag, version) of the manifest as the last sync read it
        self._lock = threading.RLock()
        self.manifest_version = 0
        self.last_sync: float | None = None
        self.last_error: str | None = None
        self.closed = False

    # ---- the object store ---------------------------------------------------------------------------------------

    def _read_manifest(self, if_changed_from=None):
        """(manifest, tag), or (None, tag) when unchanged since `if_changed_from`; an empty one when none exists."""
        obs = _obstore()
        from obstore import exceptions as ex

        try:
            opts = {"if_none_match": if_changed_from[0]} if if_changed_from and if_changed_from[0] else None
            got = obs.get(self.store, MANIFEST, options=opts) if opts else obs.get(self.store, MANIFEST)
            data = bytes(got.bytes())
            meta = got.meta
        except ex.NotModifiedError:
            return None, if_changed_from
        except (FileNotFoundError, ex.NotFoundError):
            return {"format": MANIFEST_FORMAT, "version": 0, "entries": {}}, None
        except Exception as e:
            raise TaskStoreUnavailable(f"the task store cannot be read ({type(e).__name__}: {e})") from e
        try:
            manifest = json.loads(data)
        except ValueError as e:
            raise TaskPersistError(f"{MANIFEST} in {self.label} is not JSON ({e})") from e
        if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
            raise TaskPersistError(
                f"{MANIFEST} in {self.label}: format {manifest.get('format')!r}, not {MANIFEST_FORMAT}"
            )
        return manifest, (meta.get("e_tag"), meta.get("version"))

    def _write_manifest(self, manifest: dict, tag):
        obs = _obstore()
        from obstore import exceptions as ex

        if tag is None:
            mode = "create"
        else:
            mode = {k: v for k, v in zip(("e_tag", "version"), tag) if v is not None}
        try:
            r = obs.put(self.store, MANIFEST, dumps(manifest).encode(), mode=mode)
        except (ex.PreconditionError, ex.AlreadyExistsError) as e:
            raise _Raced() from e
        except Exception as e:
            raise TaskStoreUnavailable(f"the task store cannot be written ({type(e).__name__}: {e})") from e
        return r.get("e_tag"), r.get("version")

    def _put_object(self, digest: str, data: bytes) -> None:
        obs = _obstore()
        from obstore import exceptions as ex

        try:
            obs.put(self.store, f"objects/{digest}.json", data, mode="create")
        except ex.AlreadyExistsError:
            return  # the same bytes are already there
        except Exception as e:
            raise TaskStoreUnavailable(f"the task store cannot be written ({type(e).__name__}: {e})") from e

    def _get_object(self, digest: str) -> dict:
        obs = _obstore()
        try:
            data = bytes(obs.get(self.store, f"objects/{digest}.json").bytes())
        except Exception as e:
            raise TaskStoreUnavailable(f"task object {digest[:12]} cannot be read ({type(e).__name__}: {e})") from e
        if hashlib.sha256(data).hexdigest() != digest:
            raise TaskPersistError(f"task object {digest[:12]}: its bytes do not match its digest; not applied")
        return json.loads(data)

    def _now(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).isoformat(timespec="seconds")

    # ---- the version rule ---------------------------------------------------------------------------------------

    def _check(self, kind: str, task_id: str, entry: dict | None) -> int:
        """The store's version of this task, after refusing a write that names another one."""
        intent = WRITE_INTENT.get()
        key = entry_key(kind, task_id)
        store_v = int(entry["version"]) if entry else 0
        expected = intent.replaces if intent.replaces is not None else self.applied.get(key, 0)
        if intent.replace or store_v == expected:
            return store_v
        if entry and entry.get("deleted") and expected == 0:
            return store_v  # a task removed before this replica ever saw it: nothing to overwrite
        raise TaskConflict(kind, task_id, store_v, expected)

    def version_of(self, kind: str, task_id: str):
        return self.applied.get(entry_key(kind, task_id))

    def precheck(self, kind: str, task_id: str) -> None:
        """Before an expensive registration: refuse now if this replica does not teach, or if the store already holds
        another version than the one the write names (the same check runs again when the record is published)."""
        if self.readonly:
            raise TaskStoreReadOnly(
                "this replica does not teach tasks (--task-store-readonly); send task writes to the designated replica"
            )
        manifest, _ = self._read_manifest()
        self._check(kind, task_id, manifest["entries"].get(entry_key(kind, task_id)))

    # ---- writes -------------------------------------------------------------------------------------------------

    def save(self, kind: str, record: dict) -> None:
        if self.readonly:
            raise TaskStoreReadOnly(
                "this replica does not teach tasks (--task-store-readonly); send task writes to the designated replica"
            )
        data = dumps(record).encode()
        digest = hashlib.sha256(data).hexdigest()
        self._publish(kind, str(record["id"]), digest, data, deleted=False)

    def delete(self, kind: str, task_id: str) -> None:
        if self.readonly:
            raise TaskStoreReadOnly(
                "this replica does not teach tasks (--task-store-readonly); send task writes to the designated replica"
            )
        self._publish(kind, str(task_id), None, None, deleted=True)

    def _publish(self, kind, task_id, digest, data, deleted: bool) -> None:
        key = entry_key(kind, task_id)
        with self._lock:
            if not deleted:
                self._put_object(digest, data)
            for _ in range(MAX_RACES):
                manifest, tag = self._read_manifest()
                entry = manifest["entries"].get(key)
                if deleted and (entry is None or entry.get("deleted")):
                    self.applied[key] = int(entry["version"]) if entry else 0
                    return  # nothing to remove
                version = self._check(kind, task_id, entry) + 1
                manifest["entries"][key] = {
                    "id": task_id,
                    "kind": kind,
                    "version": version,
                    "object": None if deleted else digest,
                    "deleted": deleted,
                    "updated": self._now(),
                }
                manifest["version"] = int(manifest.get("version", 0)) + 1
                self._drop_old_tombstones(manifest)
                try:
                    self._write_manifest(manifest, tag)
                except _Raced:
                    continue  # another task changed (or this one: the check above says which)
                self.applied[key] = version
                self.ids[key] = (kind, task_id)
                return
            raise TaskStoreUnavailable("the task store's manifest kept changing under this write; try again")

    def _drop_old_tombstones(self, manifest: dict) -> None:
        cutoff = self.clock() - self.tombstone_days * 86400
        for k in [
            k
            for k, e in manifest["entries"].items()
            if e.get("deleted") and datetime.fromisoformat(e["updated"]).timestamp() < cutoff
        ]:
            del manifest["entries"][k]

    # ---- reads --------------------------------------------------------------------------------------------------

    def load(self) -> dict:
        """Everything the store holds, for the start: {"readout": [...], "abstention": [...]}. Raises
        TaskStoreUnavailable when the store cannot be read (a replica does not start untaught)."""
        with self._lock:
            manifest, tag = self._read_manifest()
            out = {k: [] for k in KINDS}
            for key, e in sorted(manifest["entries"].items()):
                self.applied[key] = int(e["version"])
                self.ids[key] = (e["kind"], e["id"])
                if not e.get("deleted"):
                    out[e["kind"]].append(self._get_object(e["object"]))
            self._tag, self.manifest_version = tag, int(manifest.get("version", 0))
            self.last_sync, self.last_error = self.clock(), None
            return out

    def sync_once(self, force: bool = False) -> dict:
        """Follow the manifest: apply what is newer than this replica's versions. Returns what changed."""
        changed = {"applied": [], "removed": []}
        with self._lock:
            try:
                manifest, tag = self._read_manifest(None if force else self._tag)
                if manifest is None:
                    self.last_sync, self.last_error = self.clock(), None
                    return changed
                entries = manifest["entries"]
                for key, e in sorted(entries.items()):
                    version = int(e["version"])
                    have = self.applied.get(key, 0)
                    if version < have:
                        log.warning("task store: %s went from version %s to %s; refused", key, have, version)
                        continue
                    if version == have:
                        continue
                    if e.get("deleted"):
                        if self.applier is not None:
                            self.applier.drop(e["kind"], e["id"])
                        changed["removed"].append(e["id"])
                    else:
                        record = self._get_object(e["object"])
                        if self.applier is not None:
                            self.applier.apply(e["kind"], [record])
                        changed["applied"].append(e["id"])
                    self.applied[key] = version
                    self.ids[key] = (e["kind"], e["id"])
                for key in [k for k in self.applied if k not in entries and self.applied[k] > 0]:
                    # a tombstone dropped by the retention rule before this replica polled: the task is gone
                    kind, task_id = self.ids.get(key, (None, None))
                    if kind is not None and self.applier is not None:
                        self.applier.drop(kind, task_id)
                        changed["removed"].append(task_id)
                    del self.applied[key]
                self._tag, self.manifest_version = tag, int(manifest.get("version", 0))
                self.last_sync, self.last_error = self.clock(), None
            except TaskPersistError as e:
                self.last_error = str(e)
                raise
            return changed

    # ---- start-up and reporting ---------------------------------------------------------------------------------

    def probe(self) -> None:
        """Refuse a store that does not enforce the preconditions the conflict rule rests on: a store that ignores
        `If-Match` would make every write last-writer-wins without anyone knowing."""
        obs = _obstore()
        from obstore import exceptions as ex

        path = f"probe/{uuid.uuid4().hex}.json"

        def fail(what):
            raise TaskPersistError(
                f"--task-store {self.label}: the store {what}; decisio needs conditional writes (create-only and "
                "compare-and-swap by ETag) to keep two registrations from overwriting each other"
            )

        try:
            first = obs.put(self.store, path, b"1", mode="create")
            try:
                obs.put(self.store, path, b"2", mode="create")
                fail("accepts a create-only write over an existing object")
            except ex.AlreadyExistsError:
                pass
            tag = {k: v for k, v in (("e_tag", first.get("e_tag")), ("version", first.get("version"))) if v is not None}
            second = obs.put(self.store, path, b"3", mode=tag)
            try:
                obs.put(self.store, path, b"4", mode=tag)  # the tag of the object before the last write
                fail("accepts a write against a stale ETag")
            except ex.PreconditionError:
                pass
            del second
        except TaskPersistError:
            raise
        except NotImplementedError:
            fail("does not implement conditional writes")
        except Exception as e:
            raise TaskStoreUnavailable(
                f"--task-store {self.label}: cannot write to the store ({type(e).__name__}: {e})"
            ) from e
        finally:
            try:
                obs.delete(self.store, path)
            except Exception:
                pass

    def describe(self) -> dict:
        age = None if self.last_sync is None else round(max(0.0, self.clock() - self.last_sync), 1)
        live = sum(1 for v in self.applied.values() if v > 0)
        return {
            "backend": "object-store",
            "url": self.label,
            "readonly": self.readonly,
            "poll_seconds": self.poll_seconds,
            "manifest_version": self.manifest_version,
            "entries_applied": live,
            "last_sync_age_s": age,
            "ok": self.last_error is None and self.last_sync is not None,
            "last_error": self.last_error,
        }

    def metrics(self) -> list[str]:
        d = self.describe()
        age = d["last_sync_age_s"] if d["last_sync_age_s"] is not None else -1
        return [
            "# TYPE decisio_task_store_ok gauge",
            f"decisio_task_store_ok {1 if d['ok'] else 0}",
            "# TYPE decisio_task_store_last_sync_age_seconds gauge",
            f"decisio_task_store_last_sync_age_seconds {age}",
            "# TYPE decisio_task_store_manifest_version gauge",
            f"decisio_task_store_manifest_version {d['manifest_version']}",
        ]

    def close(self) -> None:
        self.closed = True

    # ---- garbage ------------------------------------------------------------------------------------------------

    def gc(self, older_than_days: float = 1.0, dry_run: bool = False) -> list[str]:
        """Delete the objects no manifest entry points to and that are older than `older_than_days` (an object just
        written, for a manifest write still to come, is spared). Returns the paths."""
        obs = _obstore()
        manifest, _ = self._read_manifest()
        keep = {e["object"] for e in manifest["entries"].values() if e.get("object")}
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        gone = []
        for batch in obs.list(self.store, "objects/"):
            for item in batch:
                digest = item["path"].rsplit("/", 1)[-1].removesuffix(".json")
                if digest not in keep and item["last_modified"] < cutoff:
                    gone.append(item["path"])
        if not dry_run:
            for path in gone:
                obs.delete(self.store, path)
        return gone


class TaskSync(threading.Thread):
    """Follows the manifest every `poll_seconds` (a little jittered, so replicas do not poll in step)."""

    def __init__(self, backend: ObjectBackend):
        super().__init__(name="decisio-task-sync", daemon=True)
        self.backend, self.stop_event = backend, threading.Event()
        self._warned = False

    def run(self):
        while not self.stop_event.wait(self.backend.poll_seconds * random.uniform(0.9, 1.1)):
            try:
                changed = self.backend.sync_once()
                self._warned = False
                if changed["applied"] or changed["removed"]:
                    log.info("task store: applied %s, removed %s", changed["applied"], changed["removed"])
            except Exception as e:  # the replica keeps serving what it has applied
                if not self._warned:
                    log.warning("task store: sync failed (%s); serving what is applied, will retry", e)
                    self._warned = True

    def stop(self):
        self.stop_event.set()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m decisio.serve.task_store_shared", description="the shared task store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gc", help="delete objects no manifest entry points to")
    g.add_argument("url")
    g.add_argument("--endpoint")
    g.add_argument("--older-than-days", type=float, default=1.0)
    g.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    backend = ObjectBackend(open_store(a.url, a.endpoint), sanitized(a.url))
    for path in backend.gc(a.older_than_days, a.dry_run):
        print(("would delete " if a.dry_run else "deleted ") + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
