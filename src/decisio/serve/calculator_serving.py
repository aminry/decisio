# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator in the request path (decisio.calculator; docs/calculator.md): which requests get the block, what the
response says about it, and the counts /health reports.

A server shows the block only when its model records the block format it was trained with (a tuned repository's
`defaults.calculator`, `--calculator auto`, the default); a model that records none never sees it, whatever a request
asks: such a request is answered without the block and says so in `x-decisio-warning`. A request may turn the block off
(`"calculator": false`) or bring its own amounts (`"calculator_input"`, rendered by the same code, no parsing).

Within a request, `Decision` holds what was decided and, once the state's context is known, the block itself
(`Annotation`) or why there is none (`Abstain`); `SystemOne.read_state` reads it, so every read of the request (its
questions, a certified task's own read, thinking's second read) sees the same state text. Outside a request
(registering or certifying a task) the server's setting applies to each example's state.

`Reader` is the server's reader: `rules` (the default, the rule reader alone) or `cascade` (a preview: a role tagger
loaded once at start, the rule reader and the acceptance gates; decisio.calculator.cascade), set with
`--calculator-reader`, `--calculator-tagger`, `--calculator-thresholds` and `--calculator-device`, or by a tuned
repository's `defaults.calculator_reader` (`check_config`) where no reader flag is given. A tagger that cannot be
loaded (`prepare_tagger` before the tasks, `load_reader` after the engine) never stops the start: the server reads
with the rule reader and says why (`fallback_reader`). With the rule reader every response, /health and fingerprint is
what it was before the cascade existed.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
import threading
from pathlib import Path

# the warnings of a request that asked for the block where none is shown (x-decisio-warning)
NOT_TRAINED = (
    "calculator: no block shown: this server's model was not trained with the calculator block (its repository "
    "records no defaults.calculator), or the server was started with --calculator off"
)
WITH_IMAGES = "calculator: no block shown: a request with images is read without the calculator block"
PACKED = "calculator: no block shown: --mode packed reads no calculator block"
# a request that turns the block off on a server that shows it reads its state as the model was not trained to, so no
# registered task (each fitted on reads with the block) applies to it
TASKS_OFF = (
    '"task": a request with "calculator": false applies no registered task (the tasks of this server were fitted on '
    "reads with the calculator block)"
)

_CURRENT: contextvars.ContextVar[Decision | None] = contextvars.ContextVar("decisio_calculator", default=None)


class Decision:
    """One request's calculator: `on` (the block is shown where the calculator reads the state), the `version`, the
    block made from the request's own amounts (`given`), and once decided the `annotation` (an Annotation or an
    Abstain) for `state` (the request's state object), the `reason` there is none, and a `warning`."""

    __slots__ = ("on", "version", "given", "annotation", "state", "reason", "warning", "asked", "path", "ms")

    def __init__(self, on=False, version=None, given=None, reason=None, warning=None, asked=False):
        self.on, self.version, self.given = on, version, given
        self.annotation, self.state = None, None
        self.reason, self.warning, self.asked = reason, warning, asked
        # the cascade reader's path for the state and the reading's time in ms (None: the rule reader, or no reading)
        self.path, self.ms = None, None

    def header_value(self) -> str | None:
        """x-decisio-calculator: `shown; lines=N; version=V` (with `; path=P` from the cascade reader), or
        `none; reason=R`; None where the server shows no block and the request asked for none."""
        from decisio.calculator import Annotation

        if isinstance(self.annotation, Annotation):
            path = f"; path={self.annotation.path}" if self.annotation.path else ""
            return f"shown; lines={self.annotation.lines}; version={self.annotation.version}{path}"
        if not self.on and not self.asked and self.reason in (None, "off"):
            return None
        return f"none; reason={self.reason or 'unknown'}"


def request_value(value):
    """A request's `"calculator"`: true, false or null (ValueError otherwise)."""
    if value is not None and not isinstance(value, bool):
        raise ValueError('"calculator" must be true, false or null')
    return value


def decide(server_version, want, raw_input, *, images: bool, packed: bool) -> Decision:
    """What a request gets. `server_version`: the format the server shows (None: none); `want`: the request's
    `"calculator"`; `raw_input`: its `"calculator_input"` (validated strictly here, whatever the server shows:
    decisio.calculator.CalculatorInputError, a 422, names the field)."""
    from decisio.calculator import VERSIONS, from_input

    given = None
    if raw_input is not None:
        given = from_input(raw_input, version=server_version or VERSIONS[0])
    asked = want is True or raw_input is not None
    if want is False:
        return Decision(reason="off-by-request", asked=False)
    if server_version is None:
        return Decision(
            reason="off" if not asked else "not-trained", warning=NOT_TRAINED if asked else None, asked=asked
        )
    if images:
        return Decision(reason="images", warning=WITH_IMAGES if asked else None, asked=asked)
    if packed:
        return Decision(reason="packed", warning=PACKED if asked else None, asked=asked)
    return Decision(on=True, version=server_version, given=given, asked=asked)


@contextlib.contextmanager
def deciding(decision: Decision | None):
    """Inside the block, `current()` is this request's decision."""
    token = _CURRENT.set(decision)
    try:
        yield decision
    finally:
        _CURRENT.reset(token)


def current() -> Decision | None:
    return _CURRENT.get()


class Counts:
    """/health systemone.calculator: the version shown, how many requests were read with it on, how many got a block,
    and why the others did not (one count per reason); with the cascade reader also the reader, its tagger and
    thresholds, and one count per path. Never any state text."""

    def __init__(self, version, reader: Reader | None = None):
        self.version = version
        self.reader = reader or Reader()
        self._lock = threading.Lock()
        self.requests = self.shown = self.from_input = 0
        self.none: dict[str, int] = {}
        self.paths: dict[str, int] = {}

    def note(self, decision: Decision) -> None:
        from decisio.calculator import Annotation

        with self._lock:
            if decision.on:
                self.requests += 1
            if decision.path:
                self.paths[decision.path] = self.paths.get(decision.path, 0) + 1
            if isinstance(decision.annotation, Annotation):
                self.shown += 1
                self.from_input += decision.given is not None
            elif decision.on or decision.asked or decision.reason == "off-by-request":
                r = decision.reason or "unknown"
                self.none[r] = self.none.get(r, 0) + 1

    def facts(self) -> dict:
        with self._lock:
            return {
                "version": self.version,
                "requests": self.requests,
                "shown": self.shown,
                "from_input": self.from_input,
                "none": dict(sorted(self.none.items())),
                **(
                    {**self.reader.facts(), "paths": dict(sorted(self.paths.items()))}
                    if self.reader.cascade
                    else self.reader.facts()  # {} for the rule reader, unless it stands in for a cascade
                ),
            }


# ---- the reader: rules (the default) or the cascade (a preview) ---------------------------------------------------

READERS = ("rules", "cascade")
DEVICES = ("auto", "cpu", "cuda", "mps")
THRESHOLD_KEYS = ("role", "link", "flag", "band")
# what the start line names as the thresholds' source when they come from a repository's defaults, not a file
FROM_REPOSITORY = "from the repository's defaults.calculator_reader"
_COMMIT = re.compile(r"[0-9a-f]{40}")


class Reader:
    """The server's calculator reader. `rules`: the rule reader alone (the default; nothing else is set). `cascade`:
    `tagger` (loaded once, a `decisio.calculator.cascade.Tagger`), `tagger_facts` (`name`, `revision`, `device`),
    `thresholds` (a `cascade.Thresholds`, or None: the threshold rule off, so only the rule reader's agreement and the
    stated figures accept) and the file they came from. `fallback`: on a rule reader that stands in for a cascade whose
    tagger could not be loaded, what was asked for and why it is not used (`fallback_reader`)."""

    def __init__(
        self, name="rules", tagger=None, tagger_facts=None, thresholds=None, thresholds_file=None, fallback=None
    ):
        if name not in READERS:
            raise ValueError(f"unknown calculator reader {name!r}; use one of {', '.join(READERS)}")
        if name == "cascade" and tagger is None:
            raise ValueError("the cascade reader needs a tagger")
        if fallback is not None and name != "rules":
            raise ValueError("only the rule reader stands in for a reader that could not be loaded")
        if tagger is not None and not isinstance(tagger, _LockedTagger):
            tagger = _LockedTagger(tagger)
        self.name, self.tagger, self.thresholds, self.thresholds_file = name, tagger, thresholds, thresholds_file
        self.tagger_facts = dict(tagger_facts or {"name": "unnamed", "revision": "unknown", "device": "cpu"})
        self.fallback = dict(fallback) if fallback else None

    @property
    def cascade(self) -> bool:
        return self.name == "cascade"

    def kwargs(self) -> dict:
        """What `decisio.calculator.annotate` is called with: nothing for the rule reader (its own default)."""
        if not self.cascade:
            return {}
        return {"reader": "cascade", "tagger": self.tagger, "thresholds": self.thresholds}

    def threshold_facts(self):
        t = self.thresholds
        if t is None or not t.enabled:
            return "off"
        return {"role": t.role, "link": t.link, "flag": t.flag, "band": t.band}

    def facts(self) -> dict:
        """/health: the reader, its tagger (name, revision, device) and thresholds; nothing for the rule reader, unless
        it stands in for a cascade that could not be loaded: then the reader in use and the fallback's reason."""
        if not self.cascade:
            return {"reader": self.name, "fallback": dict(self.fallback)} if self.fallback else {}
        return {
            "reader": self.name,
            "tagger": {k: self.tagger_facts.get(k) for k in ("name", "revision", "device")},
            "thresholds": self.threshold_facts(),
        }

    def fingerprint(self) -> dict | None:
        """The serving stack's record of the reader (None for the rule reader, so its fingerprint is what it was): the
        tagger's revision and the thresholds, which decide which blocks are shown."""
        if not self.cascade:
            return None
        return {
            "reader": self.name,
            "tagger": self.tagger_facts.get("revision"),
            "thresholds": self.threshold_facts(),
            # a precision other than float32 moves the tagger's probabilities: named only when set
            **({"dtype": self.tagger_facts["dtype"]} if self.tagger_facts.get("dtype") else {}),
        }

    def start_line(self) -> str:
        if not self.cascade:
            if self.fallback:
                fb = self.fallback
                return (
                    f"CALCULATOR READER rules (fallback): the cascade reader's tagger {fb['tagger']} could not be "
                    f"loaded ({fb['reason']}); this server reads states with the rule reader"
                )
            return "CALCULATOR READER rules"
        f = self.tagger_facts
        return (
            f"CALCULATOR READER cascade: tagger {f.get('name')}@{f.get('revision')} on {f.get('device')}, "
            f"thresholds {self.thresholds_file or 'off'}"
        )


def parse_tagger(spec: str) -> tuple[str, str | None]:
    """`--calculator-tagger`: (a local directory, None), or (a Hub repository, its full commit hash) from
    `owner/name@<commit>`. ValueError otherwise: a Hub tagger at a branch, a tag or no revision could change."""
    if Path(spec).is_dir():
        return spec, None
    name, at, revision = spec.rpartition("@")
    if not at or not name or "/" not in name:
        raise ValueError(
            f"--calculator-tagger {spec!r}: not a directory; a tagger on the Hub is owner/name@<full commit hash>"
        )
    if not _COMMIT.fullmatch(revision):
        raise ValueError(
            f"--calculator-tagger {spec!r}: a tagger on the Hub needs a full 40-character commit hash after @, so the "
            "weights cannot change"
        )
    return name, revision


def load_thresholds(source):
    """The cascade reader's thresholds: from `--calculator-thresholds` (a JSON file) or a repository's
    `defaults.calculator_reader.thresholds` (a dict, already checked by `check_config`). An object with `role`, `link`
    and `flag` (and `band`, 0 when absent; other keys are ignored), checked as `cascade.Thresholds` checks them; None:
    the threshold rule off. ValueError: a bad file or object."""
    if source is None:
        return None
    from decisio.calculator.cascade import Thresholds

    if isinstance(source, dict):
        data, where = source, "defaults.calculator_reader.thresholds"
    else:
        where = f"--calculator-thresholds {source}"
        try:
            data = json.loads(Path(source).read_text())
        except (OSError, ValueError) as e:
            raise ValueError(f"{where}: not a readable JSON file ({type(e).__name__})") from e
    missing = [k for k in ("role", "link", "flag") if not isinstance(data, dict) or data.get(k) is None]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(missing)} (a number each)")
    try:
        return Thresholds(**{k: float(data[k]) for k in THRESHOLD_KEYS if k in data})
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: {e}") from e


def thresholds_label(source) -> str | None:
    """What the start line names as the thresholds' source: the file, the repository, or None (off)."""
    if source is None:
        return None
    return FROM_REPOSITORY if isinstance(source, dict) else str(source)


def _is_path(spec: str) -> bool:
    """A tagger named as a local directory: an existing one, or a path that says it is one (absolute, ./ or ../), so a
    repository's file can name a directory that is not on the machine that checks it."""
    return Path(spec).is_dir() or spec.startswith(("/", "./", "../"))


def check_config(value, where: str = "defaults.calculator_reader") -> list[str]:
    """A tuned repository's `defaults.calculator_reader`, checked for its form only (nothing is loaded or downloaded):
    `{"reader": "rules"}`, or `{"reader": "cascade", "tagger": ..., "thresholds": {...}}` with a tagger that is
    `owner/name@<full 40-character commit hash>` on the Hub or a local directory path, and all four thresholds (`role`,
    `link`, `flag`, `band`) as numbers from 0 to 1 that `cascade.Thresholds` accepts. ValueError names the field.
    Returns the keys it does not know (a newer decisio's), which are ignored."""
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be an object: {{"reader": "cascade", "tagger": ..., "thresholds": {{...}}}}')
    reader = value.get("reader")
    if reader not in READERS:
        raise ValueError(f'{where}.reader must be "rules" or "cascade", not {reader!r}')
    ignored = sorted(set(value) - {"reader", "tagger", "thresholds"})
    if reader == "rules":
        extra = [k for k in ("tagger", "thresholds") if k in value]
        if extra:
            raise ValueError(f"{where}: {' and '.join(extra)} belong to the cascade reader, and reader is rules")
        return ignored
    tagger = value.get("tagger")
    if not isinstance(tagger, str) or not tagger.strip():
        raise ValueError(
            f"{where}.tagger must name the cascade reader's tagger: owner/name@<full commit hash> on the Hugging Face "
            "Hub, or a local directory"
        )
    if not _is_path(tagger):
        name, at, revision = tagger.rpartition("@")
        if not at or not name or "/" not in name or not _COMMIT.fullmatch(revision):
            raise ValueError(
                f"{where}.tagger {tagger!r}: a tagger on the Hub is owner/name@<full 40-character commit hash>, so "
                "the weights cannot change; a local tagger is a directory path (absolute, or starting with ./ or ../)"
            )
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{where}.thresholds must be an object with role, link, flag and band (numbers from 0 to 1)")
    missing = [k for k in THRESHOLD_KEYS if k not in thresholds]
    if missing:
        raise ValueError(f"{where}.thresholds: missing {', '.join(missing)} (a number from 0 to 1 each)")
    for k in THRESHOLD_KEYS:
        v = thresholds[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise ValueError(f"{where}.thresholds.{k} must be a number from 0 to 1, not {v!r}")
    from decisio.calculator.cascade import Thresholds

    try:
        Thresholds(**{k: float(thresholds[k]) for k in THRESHOLD_KEYS})
    except ValueError as e:
        raise ValueError(f"{where}.thresholds: {e}") from e
    return ignored + [f"thresholds.{k}" for k in sorted(set(thresholds) - set(THRESHOLD_KEYS))]


def _available(device: str) -> bool:
    import torch

    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def tagger_device(choice: str, backend: str) -> str:
    """`--calculator-device`: auto is the serving card for the vLLM backend on CUDA, else the CPU; a device that is not
    there is a ValueError."""
    if choice not in DEVICES:
        raise ValueError(f"--calculator-device must be one of {', '.join(DEVICES)}")
    if choice == "auto":
        return "cuda" if backend == "vllm" and _available("cuda") else "cpu"
    if not _available(choice):
        raise ValueError(f"--calculator-device {choice}: no {choice} device here")
    return choice


def local_revision(directory) -> str:
    """A local tagger directory's identity: the SHA-256 of its SHA256SUMS (checked against the files at load), else of
    the list of every file's path and SHA-256."""
    d = Path(directory)
    sums = d / "SHA256SUMS"
    if sums.is_file():
        return "sha256:" + hashlib.sha256(sums.read_bytes()).hexdigest()
    lines = []
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(d).as_posix()}")
    return "sha256:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()


def reader_identity(reader: str, tagger, thresholds) -> str | None:
    """What the task fingerprint names for the cascade reader (None for the rule reader): the tagger's revision (a Hub
    commit, or a local directory's `local_revision`) and the thresholds' SHA-256 (the file's bytes, or the canonical
    JSON of a repository's four values), so that a task fitted on reads of one reader does not apply under another."""
    if reader != "cascade" or not tagger:
        return None
    tag = str(tagger)
    rev = tag.rsplit("@", 1)[1] if "@" in tag and not Path(tag).is_dir() else local_revision(tag)
    if isinstance(thresholds, dict):
        raw = json.dumps({k: float(thresholds[k]) for k in THRESHOLD_KEYS if k in thresholds}, sort_keys=True).encode()
    else:
        raw = Path(thresholds).read_bytes() if thresholds else None
    thr = "sha256:" + hashlib.sha256(raw).hexdigest()[:16] if raw is not None else "off"
    return f"cascade@{rev};thresholds={thr}"


class _LockedTagger:
    """A tagger called by one request at a time: concurrent requests read through one model in the server process."""

    def __init__(self, tagger):
        self._tagger, self._lock = tagger, threading.Lock()

    def predict(self, state, context=None, during=None):
        """The tagger's predict under the lock; `during` (the CPU work the cascade runs while a GPU encoder computes)
        is passed on to a tagger that takes it (one that says `overlaps`), else run first, so the wrapper keeps the
        tagger's own call."""
        with self._lock:
            if during is None:
                return self._tagger.predict(state, context)
            if getattr(self._tagger, "overlaps", False):
                return self._tagger.predict(state, context, during=during)
            during()
            return self._tagger.predict(state, context)

    def __getattr__(self, name):
        return getattr(self._tagger, name)


def check_flags(reader: str, tagger, thresholds, device: str) -> str | None:
    """The reader flags' start check, before anything is loaded: an error message (the start stops), or None. The
    tagger flags belong to the cascade; the cascade needs a tagger; the tagger's form and the thresholds file are
    checked here. A cascade on a server that shows no block is not an error (`not_used_note`)."""
    if reader not in READERS:
        return f"--calculator-reader must be one of {', '.join(READERS)}"
    if reader == "rules":
        given = [
            flag
            for flag, value in (
                ("--calculator-tagger", tagger),
                ("--calculator-thresholds", thresholds),
                ("--calculator-device", None if device == "auto" else device),
            )
            if value is not None
        ]
        if given:
            return f"{', '.join(given)}: for --calculator-reader cascade (this server reads with the rule reader)"
        return None
    if not tagger:
        return "--calculator-reader cascade needs --calculator-tagger (a local directory or owner/name@<commit>)"
    try:
        parse_tagger(tagger)
        load_thresholds(thresholds)
    except ValueError as e:
        return str(e)
    return None


def not_used_note(reader: str, calculator_version) -> str | None:
    """The start note of a cascade reader on a server that shows no calculator block (its model records none,
    `--calculator off`, or packed mode): nothing is loaded and the server reads no state."""
    if reader == "cascade" and not calculator_version:
        return (
            "CALCULATOR READER cascade: not used: this server shows no calculator block (its model records no "
            "defaults.calculator, --calculator off, or --mode packed), so no tagger is loaded"
        )
    return None


def prepare_tagger(tagger, device: str, backend: str, resolve=None) -> Path:
    """The tagger's files, made ready at start before the tasks are loaded, so that a tagger that cannot be used is
    known while the task fingerprint can still name the reader in use: the spec's form, the device, torch and
    transformers importable, the files on disk (a Hub tagger downloaded at its commit by `resolve`, default
    `cascade.tagger.resolve`), their SHA256SUMS and the tagger's format. Returns the directory; raises otherwise (any
    exception: the caller falls back to the rule reader)."""
    from decisio.calculator.cascade import tagger as tagger_module

    name, revision = parse_tagger(tagger)
    tagger_device(device, backend)  # imports torch; a device that is not there is a ValueError
    directory = Path((resolve or tagger_module.resolve)(name, revision))
    tagger_module.check_sums(directory)
    try:
        meta = tagger_module.read_meta(directory)
    except (OSError, ValueError) as e:
        raise ValueError(f"tagger directory: no readable cascade_tagger.json ({type(e).__name__})") from e
    from decisio.calculator.cascade.model import FORMAT, LABELS  # torch

    if meta.get("format") != FORMAT or tuple(meta.get("labels", ())) != LABELS:
        raise ValueError("tagger directory: an unknown format or label set")
    import transformers  # noqa: F401  (the encoder's code: a missing install is known here, not after the engine)

    return directory


def fallback_reason(error: BaseException) -> str:
    """One line on why a tagger could not be loaded, for the start line and /health."""
    text = " ".join(str(error).split()) or "no message"
    if isinstance(error, ImportError):
        text = f"{text}; the cascade reader needs torch and transformers"
    text = f"{type(error).__name__}: {text}"
    return text if len(text) <= 300 else text[:297] + "..."


def fallback_reader(tagger, error: BaseException) -> Reader:
    """The rule reader, standing in for a cascade whose tagger could not be loaded: it serves as the default does, and
    its start line and /health name the tagger and the reason."""
    return Reader("rules", fallback={"wanted": "cascade", "tagger": str(tagger), "reason": fallback_reason(error)})


def load_reader(
    reader: str,
    tagger,
    thresholds,
    device: str,
    backend: str,
    loader=None,
    dtype: str = "float32",
    directory=None,
) -> Reader:
    """The server's reader, loaded once at start (after `check_flags`): the rule reader, or the cascade with its tagger
    on the resolved device. `loader(path, revision, device)` loads the tagger (default: `cascade.tagger.HFTagger.load`,
    which needs torch and transformers and checks a directory's SHA256SUMS). `directory`: the files `prepare_tagger`
    made ready, loaded from there (the tagger keeps its name and revision); `thresholds`: a file or a repository's
    dict."""
    if reader == "rules":
        return Reader()
    from decisio.calculator.cascade.tagger import HFTagger

    name, revision = parse_tagger(tagger)
    dev = tagger_device(device, backend)
    path, at = (str(directory), None) if directory is not None else (name, revision)
    load = loader or HFTagger.load
    model = load(path, at, dev, dtype) if dtype != "float32" else load(path, at, dev)
    if revision is None:  # a local directory: its content; a Hub tagger: the commit hash
        revision = local_revision(name)
    facts = {"name": name, "revision": revision, "device": dev, **({"dtype": dtype} if dtype != "float32" else {})}
    return Reader("cascade", model, facts, load_thresholds(thresholds), thresholds_label(thresholds))
