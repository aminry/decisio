# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Where taught tasks are kept so that a restart does not lose them (docs/tasks.md, "Saved to disk").

A task store holds two kinds of record: `readout` tasks (`POST /v1/tasks`, decisio.serve.tasks) and `abstention` tasks
(`POST /v1/abstention/tasks`, decisio.serve.abstention). A backend keeps both, one record per task, and has three
operations:

    backend.save(kind, record)      write one task record durably, replacing the task of the same id
    backend.delete(kind, task_id)   remove it
    backend.load()                  {"readout": [record, ...], "abstention": [record, ...]}

A write that fails raises `TaskPersistError`, and the caller does not report the task as registered (or removed).

`DirBackend` is the first backend: one JSON file per task in a directory (`--tasks-dir`). A record is the form
`GET /v1/tasks?full=1` returns, so a file can be put into a `--tasks-file` list as it is. Every write goes to a
temporary file in the same directory, is flushed and fsync'ed, then renamed over the task's file, and the directory is
fsync'ed, so a crash leaves the previous file or the new one, never a half-written one. One server owns a directory
(an exclusive lock on `<dir>/.lock`): a second server on it does not start. A file that cannot be read stops the start,
naming the file: a lost or damaged store must not become untaught answers silently.

Standard library only.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KINDS = ("readout", "abstention")


class TaskPersistError(RuntimeError):
    """A task could not be written (or removed) durably; nothing was registered (or removed)."""


class TaskStoreUnavailable(TaskPersistError):
    """The shared store cannot be reached: a write fails (503), serving goes on with what is applied."""


class TaskStoreReadOnly(TaskPersistError):
    """This replica does not teach tasks (`--task-store-readonly`); the change belongs on the designated replica."""


class TaskConflict(TaskPersistError):
    """The shared store holds another version of this task than the one the write names (409)."""

    def __init__(self, kind: str, task_id: str, store_version: int, your_version: int):
        self.kind, self.task_id, self.store_version, self.your_version = kind, task_id, store_version, your_version
        super().__init__(
            f"{kind} task {task_id!r}: the shared store holds version {store_version}, not version {your_version}: "
            "someone taught it since; read it again and send it again, or send ?replace=1 to replace whatever the "
            "store holds"
        )


@dataclass
class WriteIntent:
    """What a caller of a task write says about the shared store: `replace` ignores the version check, `replaces` names
    the version it was shown (None: the version this replica serves). Set per request by the routes."""

    replace: bool = False
    replaces: int | None = None


WRITE_INTENT: contextvars.ContextVar[WriteIntent] = contextvars.ContextVar("task_write_intent", default=WriteIntent())


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(f"{type(x).__name__} is not JSON serialisable")


def dumps(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=jsonable)


class DirBackend:
    """One JSON file per task in `path`; see the module's account."""

    kind_name = "dir"

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise TaskPersistError(f"--tasks-dir {self.path}: cannot create the directory ({e})") from e
        self._hold = self._acquire()
        for stray in self.path.glob(".tmp-*"):  # a write that never reached its rename
            try:
                stray.unlink()
            except OSError:
                pass

    # ---- ownership -----------------------------------------------------------------------------------------------

    def _acquire(self):
        try:
            import fcntl
        except ImportError:  # not POSIX: no lock, the single-server rule is the operator's
            return None
        try:
            fh = open(self.path / ".lock", "a+")
        except OSError as e:
            raise TaskPersistError(f"--tasks-dir {self.path}: cannot open the lock file ({e})") from e
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            raise TaskPersistError(
                f"--tasks-dir {self.path} is in use by another server: one directory belongs to one server "
                "(replicas share a store, not a directory)"
            ) from e
        return fh

    def close(self):
        if self._hold is not None:
            self._hold.close()
            self._hold = None

    # ---- records -------------------------------------------------------------------------------------------------

    def file_for(self, kind: str, task_id: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"unknown task kind {kind!r}")
        # the id is the customer's string: the file is named by its digest, the id is inside the record
        return self.path / f"{kind}-{hashlib.sha256(str(task_id).encode()).hexdigest()[:32]}.json"

    def save(self, kind: str, record: dict) -> None:
        target = self.file_for(kind, record["id"])
        try:
            data = dumps(record).encode()
        except (TypeError, ValueError) as e:
            raise TaskPersistError(f"task {record.get('id')!r}: not serialisable ({e})") from e
        tmp = self.path / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
        with self._lock:
            try:
                with open(tmp, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
                self._sync_dir()
            except OSError as e:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise TaskPersistError(f"task {record.get('id')!r}: not saved to {self.path} ({e})") from e

    def delete(self, kind: str, task_id: str) -> None:
        target = self.file_for(kind, task_id)
        with self._lock:
            try:
                target.unlink()
                self._sync_dir()
            except FileNotFoundError:
                return
            except OSError as e:
                raise TaskPersistError(f"task {task_id!r}: not removed from {self.path} ({e})") from e

    def _sync_dir(self) -> None:
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return  # a platform that cannot open a directory: the rename is as durable as it gets
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def load(self) -> dict:
        out = {k: [] for k in KINDS}
        for kind in KINDS:
            for f in sorted(self.path.glob(f"{kind}-*.json")):
                try:
                    record = json.loads(f.read_text())
                except (OSError, ValueError) as e:
                    raise TaskPersistError(
                        f"{f}: cannot be read as a task ({e}); move it away to start without it, or restore it"
                    ) from e
                if not isinstance(record, dict) or not record.get("id"):
                    raise TaskPersistError(f"{f}: not a task record (no id)")
                out[kind].append(record)
        return out

    def describe(self) -> dict:
        return {"backend": "dir", "path": str(self.path)}

    # the shared store's hooks, which a directory does not need
    def precheck(self, kind: str, task_id: str) -> None:
        return None

    def version_of(self, kind: str, task_id: str):
        return None


def load_at_start(store, backend, file_tasks=None, file_abstention=None, saved=None):
    """Fill `store` at start; returns (the abstention task records to serve, the report of what loaded from where).

    No backend: the `--tasks-file` records load into the store (as before) and the `--abstention-tasks` records are
    served, nothing is saved. With a backend (`--tasks-dir`) it is the store of record: its tasks load first, all of
    them (a task fitted under another stack stays loaded and is not applied, as any task of another stack); then each
    record of `--tasks-file` / `--abstention-tasks` whose id the backend does not hold is saved to it (the file seeds
    the directory), and one whose id it holds is left as the backend has it, because a restart must not undo a task
    taught since. To replace a task from a file, import it (`POST /v1/tasks/import`) or delete it first."""
    report = {"saved_to": backend.describe() if backend is not None else None}
    file_tasks, file_abstention = list(file_tasks or []), list(file_abstention or [])
    if backend is None:
        store.load(file_tasks)
        report["from_file"] = {"readout": len(file_tasks), "abstention": len(file_abstention)}
        return file_abstention, report
    store.backend = backend
    saved = saved if saved is not None else backend.load()
    store.load(saved["readout"], persist=False)
    have = {t["id"] for t in store.by_key.values()}
    # a task another replica seeded first, or a replica that does not teach, leaves the seed unsaved (not an error)
    unsaved = []
    seed = []
    for t in (t for t in file_tasks if t.get("id") not in have):
        try:
            store.load([t])
            seed.append(t)
        except (TaskConflict, TaskStoreReadOnly):
            unsaved.append(t.get("id"))
    abstention = {t["id"]: t for t in saved["abstention"]}
    seed_a = []
    for t in (t for t in file_abstention if t.get("id") not in abstention):
        try:
            backend.save("abstention", t)
            abstention[t["id"]] = t
            seed_a.append(t)
        except (TaskConflict, TaskStoreReadOnly):
            unsaved.append(t.get("id"))
    held = "from_dir" if backend.describe().get("backend") == "dir" else "from_store"
    report[held] = {"readout": len(saved["readout"]), "abstention": len(saved["abstention"])}
    report["from_file"] = {"readout": len(seed), "abstention": len(seed_a)}
    if unsaved:
        report["file_tasks_not_saved"] = unsaved
    kept = [t.get("id") for t in file_tasks if t.get("id") in have] + [
        t.get("id") for t in file_abstention if t.get("id") in {x["id"] for x in saved["abstention"]}
    ]
    if kept:
        report["file_tasks_left_as_saved"] = kept
    return list(abstention.values()), report


def usable(store, abstention_records, served_name, layout="plain") -> int:
    """How many loaded tasks apply to this server: readout tasks of its stack, abstention tasks of its served model and
    layout."""
    from decisio.names import same_fingerprint

    n = sum(1 for t in store.by_key.values() if same_fingerprint(t["fingerprint"], store.fingerprint))
    return n + sum(
        1
        for t in abstention_records
        if t.get("model") in (None, served_name) and (t.get("layout") or "plain") == layout
    )


def start_refusal(usable_tasks: int, required: int | None, loaded: int | None, where_held=None) -> str | None:
    """Why the server must not start (`--require-tasks`), or None. `where_held`: "the directory /x", "the task store
    s3://b/p"; `loaded`: how many tasks it held."""
    if required is None or usable_tasks >= required:
        return None
    where = f" ({where_held} held {loaded})" if where_held is not None and loaded is not None else ""
    return (
        f"--require-tasks {required}: {usable_tasks} task(s) apply to this server{where}: a lost or empty store would "
        "otherwise serve untaught answers. Restore the store, or lower the number."
    )
