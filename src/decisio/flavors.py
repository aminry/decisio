# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Flavors: named configurations of the same weights (docs/design/flavors.md).

    calibrated        calibrated probabilities for every type, no thinking, no commit rule, no read twice
    commit            commits yes/no answers outside the uncertain middle band; calibrated probabilities, no thinking
    read-enhanced     reads short requests twice, with the threshold and the question cap, no thinking

`--flavor NAME` applies one to a base and names the server `Decisio-31b-calibrated`, `Decisio-12b-commit`,
`Decisio-12b-fp8-read-enhanced` (the 31B is served in FP8 only, so only the 12B's names say which precision).
Every flavor uses the header layout.
An explicit flag wins over the flavor, on any flavor; each such flag is recorded as an override (`/health`,
`x-decisio-flavor`, the start line, the serving-stack fingerprint).
A flavor whose defining value is missing refuses to start; one with a missing secondary value serves marked `pending`.
A tuned model's repository can be a flavor too (`flavor` and `defaults` in its decision_config.json, `from_repository`):
the same machinery names the server, records the overrides and puts the flavor in the fingerprint.
With no flavor named a server is exactly what it was.

    f = flavor_of("gemma-4-31b", "read-enhanced")
    apply_settings(args, f)            # layout, commit, thinking, read twice: before the prompt format is built
    apply_temperatures(args, f)        # yes/no, score, choice: after the flags and the temperatures file
    report = flavor_report(f, snapshot, args)

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FLAVOR_NAMES = ("calibrated", "commit", "read-enhanced")
# a tuned repository can also be a flavor the bases do not have: the commit rule with thinking on the question types its
# settings name (`defaults.think`), which a tuned model's training shows where it helps
TUNED_FLAVOR_NAMES = (*FLAVOR_NAMES, "commit-think")

# the served-name stem of each base that has flavors: the 31B is FP8 only; the 12B has two precisions
STEMS = {"gemma-4-31b": "Decisio-31b", "gemma-4-12b": "Decisio-12b", "gemma-4-12b-fp8": "Decisio-12b-fp8"}

# the arguments a flavor sets, as the server's argument names; the override check reads exactly these
SETTINGS = (
    "layout",
    "noul_commit",
    "think",
    "think_factor",
    "think_budget",
    "think_types",
    "read_twice_under",
    "read_twice_max_questions",
)
TEMPERATURE_ARGS = ("temperature", "temperature_choice", "temperature_noul", "temperature_score", "temperatures_file")
THINK_TEMPERATURE_ARGS = ("think_temperature_choice", "think_temperature_noul", "think_temperature_score")
SNAPSHOT = SETTINGS + TEMPERATURE_ARGS + THINK_TEMPERATURE_ARGS + ("served_name",)


class FlavorError(ValueError):
    """A flavor that cannot be served: unknown name, not for this base, or a defining value still missing."""


@dataclass(frozen=True)
class Flavor:
    name: str  # calibrated | commit | read-enhanced
    stem: str  # Decisio-31b | Decisio-12b | Decisio-12b-fp8
    layout: str = "header"
    noul_commit: bool = False
    think: bool = False
    think_factor: float | None = None  # None: the server's default
    think_budget: int | None = None
    think_types: list | None = None  # the question types that think; None: all
    read_twice_under: int = 0
    read_twice_max_questions: int = 0
    temperatures: dict = field(default_factory=dict)  # yes/no and score; empty: the base's profile
    choice_temperature: float | None = None
    choice_bands: dict = field(default_factory=dict)
    think_temperatures: dict = field(default_factory=dict)
    # a value that is a placeholder: the flavor serves and says so (/health, x-decisio-flavor, the start line)
    pending: tuple[str, ...] = ()
    # a defining value that does not exist yet: the flavor refuses to start, with this as the reason
    unavailable: str | None = None

    @property
    def served_name(self) -> str:
        return f"{self.stem}-{self.name}"

    def settings(self) -> dict:
        """What the flavor sets, as the server's argument names (None for what it leaves to the server's default)."""
        out = {k: getattr(self, k) for k in SETTINGS}
        out["noul_commit"], out["think"] = self.noul_commit, self.think
        return out


# The read-twice threshold is the measured choice of #296 on the 31B (N = 2,000 counted input tokens); R, the question
# cap, waits for a timing run on the target card. On the 12B read twice added nothing within noise: the flavor carries
# it and says so.
READ_TWICE_N = 2000


# The calibrated temperatures of the flavors, per base: yes/no and score (choice keeps the base's own).
# Fitted by minimum log loss on the engine's readouts read under the header layout, and judged on calibration by
# question type, not on any scoring rule. Yes/no: a private 1,400-item suite, 5-fold cross-validation stratified by task
# (out of fold, the 12B's log loss goes 0.3225 -> 0.3062 and ECE 0.064 -> 0.058); each precision on its own readouts,
# and only on items that share no text with a public benchmark's test set (599 of the suite's 600 yes/no items).
# Score: a private natural calibration split and a 500-item development set that share none either (the suite has no
# score questions). Choice did not move: 3.52 fitted against the base's 3.592, flat out of fold, and the suite does not
# support option-count bands.
# A base with no entry has not been read yet: its `calibrated` and `commit` refuse to start (the 31B waits for its own
# read of the suite).
CALIBRATED = {
    "gemma-4-12b": {"noul": 4.84, "score": 7.85},
    "gemma-4-12b-fp8": {"noul": 4.72, "score": 7.85},
}
NOT_FITTED = (
    "the calibrated temperatures are not fitted yet: they wait for the 1,400-item suite read under the header on this "
    "base (docs/design/flavors.md)"
)


def _table() -> dict:
    out = {}
    for base, stem in STEMS.items():
        is31 = base == "gemma-4-31b"
        temps = dict(CALIBRATED.get(base, {}))
        miss = None if temps else NOT_FITTED
        out[base] = {
            "calibrated": Flavor("calibrated", stem, temperatures=temps, unavailable=miss),
            "commit": Flavor("commit", stem, noul_commit=True, temperatures=temps, unavailable=miss),
            "read-enhanced": Flavor(
                "read-enhanced",
                stem,
                read_twice_under=READ_TWICE_N,
                temperatures=temps,
                pending=(
                    *(
                        ()
                        if temps
                        else ("temperatures: the calibrated ones are not fitted yet, the base's own are served",)
                    ),
                    "read_twice_max_questions: the question cap waits for the timing cell",
                    *(() if is31 else ("read twice is measured on the 31B; on the 12B it added nothing within noise",)),
                ),
            ),
        }
    return out


FLAVORS = _table()


def from_repository(repo) -> Flavor:
    """The flavor a tuned repository is (its `flavor` block and `defaults`, decisio.hub): the values come from the
    repository, not from the table above, because a tuned model's temperatures are its own. The repository's checks
    (`hub.check_flavor`) have already made sure `defaults` are that flavor's."""
    block, d = repo.flavor, repo.defaults
    name, served = block["name"], block["served_name"]
    temps = d["temperatures"]
    think = d.get("think") or {}
    return Flavor(
        name,
        served[: -len(name) - 1],
        noul_commit=bool(d["noul_commit"]),
        think=bool(think),
        think_factor=think.get("factor"),
        think_budget=think.get("budget"),
        think_types=think.get("types"),
        read_twice_under=int(d.get("read_twice_under") or 0),
        read_twice_max_questions=int(d.get("read_twice_max_questions") or 0),
        temperatures={k: float(temps[k]) for k in ("noul", "score") if k in temps},
        choice_temperature=float(temps["choice"]) if "choice" in temps else None,
        choice_bands={b: float(t) for b, t in (d.get("choice_bands") or {}).items()},
    )


def served_names(base_key: str) -> list[str]:
    """The three served names of a base, capitalised as they are reported."""
    stem = STEMS.get(base_key)
    return [f"{stem}-{n}" for n in FLAVOR_NAMES] if stem else []


def flavor_of(base_key: str, name: str) -> Flavor:
    """The flavor `name` of a base, refused when there is none for the base or its defining value is missing."""
    if name not in FLAVOR_NAMES:
        raise FlavorError(f"--flavor {name!r} is not one of {', '.join(FLAVOR_NAMES)}")
    if base_key not in FLAVORS:
        raise FlavorError(f"--flavor {name} is for the bases {', '.join(sorted(FLAVORS))}; {base_key} has no flavors")
    f = FLAVORS[base_key][name]
    if f.unavailable:
        raise FlavorError(f"--flavor {name} on {base_key}: {f.unavailable}")
    return f


def snapshot(args) -> dict:
    """What the command line gave, before any default or flavor is applied (the override check compares against it)."""
    return {k: getattr(args, k, None) for k in SNAPSHOT}


def apply_settings(args, f: Flavor) -> None:
    """The flavor's layout, commit rule, thinking and read twice, where the command line gave none; before the prompt
    format is built and before the repository's defaults and the server's own defaults fill what is left."""
    for k, v in f.settings().items():
        if getattr(args, k, None) is None and v is not None:
            setattr(args, k, v)
    if getattr(args, "served_name", None) is None:
        args.served_name = f.served_name


def apply_temperatures(args, f: Flavor) -> None:
    """The flavor's temperatures where nothing gave one (a flag or the temperatures file come first); the base's
    profile fills the rest after this."""
    for qtype in ("noul", "score"):
        if getattr(args, f"temperature_{qtype}", None) is None and qtype in f.temperatures:
            setattr(args, f"temperature_{qtype}", float(f.temperatures[qtype]))
    if getattr(args, "temperature_choice", None) is None and not getattr(args, "choice_bands", None):
        if f.choice_bands:
            args.choice_bands = {b: float(t) for b, t in f.choice_bands.items()}
        elif f.choice_temperature is not None:
            args.temperature_choice = float(f.choice_temperature)
    for qtype, t in f.think_temperatures.items():
        if getattr(args, f"think_temperature_{qtype}", None) is None:
            setattr(args, f"think_temperature_{qtype}", float(t))


def overrides(f: Flavor, given: dict) -> dict:
    """{argument: {"flavor": the flavor's value, "served": what the command line gave}} for every flag the command line
    gave (`given`, from `snapshot`) that is not what the flavor sets. A flag the flavor does not set is an override
    too: the server is not what the name says."""
    out = {}
    for k in SETTINGS:
        v = given.get(k)
        if v is not None and v != getattr(f, k):
            out[k] = {"flavor": getattr(f, k), "served": v}
    for k in TEMPERATURE_ARGS + THINK_TEMPERATURE_ARGS:
        v = given.get(k)
        if v is None:
            continue
        mine = None
        if k.startswith("temperature_") and k[len("temperature_") :] in ("noul", "score"):
            mine = f.temperatures.get(k[len("temperature_") :])
        elif k == "temperature_choice":
            mine = f.choice_temperature
        elif k.startswith("think_temperature_"):
            mine = f.think_temperatures.get(k[len("think_temperature_") :])
        if v != mine:
            out[k] = {"flavor": mine, "served": v}
    if given.get("served_name") not in (None, f.served_name):
        out["served_name"] = {"flavor": f.served_name, "served": given["served_name"]}
    return out


def flavor_report(f: Flavor, given: dict) -> dict:
    """What /health, the start line and the serving-stack fingerprint say of the flavor."""
    return {
        "name": f.name,
        "served_name": f.served_name,
        "overrides": overrides(f, given),
        "pending": list(f.pending),
    }


def header_value(report: dict) -> str:
    """The x-decisio-flavor value: the short name, with `+overrides` and `+pending` where they apply."""
    return report["name"] + ("+overrides" if report["overrides"] else "") + ("+pending" if report["pending"] else "")


def fingerprint_facts(report: dict) -> dict:
    """The part of the serving-stack fingerprint a flavor adds (only when one is set): the name, the overrides' values
    and which values are placeholders, so a certificate made under another flavor, other overrides or placeholder
    values is suspended."""
    return {
        "name": report["name"],
        "overrides": {k: v["served"] for k, v in sorted(report["overrides"].items())},
        "pending": len(report["pending"]),
    }


def siblings(report: dict) -> list[str]:
    """The other flavors' served names of the same base (a request naming one is refused)."""
    stem = report["served_name"][: -len(report["name"]) - 1]
    names = TUNED_FLAVOR_NAMES if report["name"] in TUNED_FLAVOR_NAMES[len(FLAVOR_NAMES) :] else FLAVOR_NAMES
    return [f"{stem}-{n}" for n in names if n != report["name"]]
