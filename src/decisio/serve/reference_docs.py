# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Reference documents generated from the code: an OpenAPI file for the served routes and a table of the server's flags.

    python -m decisio.serve.reference_docs --write [--out docs/reference]     # write openapi.json and flags.md
    python -m decisio.serve.reference_docs --check [--out docs/reference]     # exit 1 if the files in --out are not what the code gives

`openapi.json` is FastAPI's own document for the real app (`make_app` with `add_routes`, built over a stub engine, so the
routes, paths, methods and parameters are the server's), with the request and response models added to it: the models
below describe the wire (`docs/api.md` is the prose), and `tests/unit/test_reference_docs.py` checks real responses of the
route against them. `flags.md` is the table of `decisio.serve.vllm_engine.build_parser()`: every flag, its default, its
values and its help text.

The two files carry the release's version (`info.version`, and a marker line in `flags.md`); release-please rewrites both
in the release pull request (`release-please-config.json`, `extra-files`), so they are written at release time and the
site renders them from the tag. A test fails when the files and the code disagree, so a change to a route or a flag
brings its regenerated files with it.
"""

# ruff: noqa: E501  (descriptions of the wire, kept on one line each so a diff of the generated file reads against them)
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

import decisio
from decisio.serve.systemone import SystemOneRequest

CERTIFIED_REASONS = (
    "below_threshold",
    "nothing_certified",
    "serving_stack_changed",
    "traffic_changed",
    "state_truncated",
    "header_limited",
    "think_mode_changed",
    "concurrency_changed",
)


# ---- the wire, as models ----------------------------------------------------------------------------------------------


class Usage(BaseModel):
    input_tokens: int = Field(
        description="The state and each question's own text in the served tokenizer, plus the image tokens."
    )
    output_tokens: int = Field(
        description="One per question, plus the reasoning tokens of the questions that thought (`thinking_tokens`)."
    )
    thinking_tokens: int | None = Field(
        None, description="The reasoning tokens generated for the request; present when thinking was on for it."
    )


class _Answer(BaseModel):
    model_config = ConfigDict(extra="allow")
    certified: bool | None = Field(
        None, description="Present when a registered certified task applies: whether this answer carries the promise."
    )
    certificate: str | None = Field(None, description="The certificate's id, with `certified`.")
    certified_reason: (
        Literal[
            "below_threshold",
            "nothing_certified",
            "serving_stack_changed",
            "traffic_changed",
            "state_truncated",
            "header_limited",
            "think_mode_changed",
            "concurrency_changed",
        ]
        | None
    ) = Field(None, description="Why `certified` is false.")
    unknown_probability: float | None = Field(
        None, description="imajev's extension (its encodings only): the probability of an unknown outcome."
    )
    abstained: bool | None = Field(
        None, description="imajev's extension, and abstention tasks: whether the server abstained."
    )


class NoulAnswer(_Answer):
    type: Literal["noul"]
    noul: float = Field(description="The probability of yes.")


class ChoiceAnswer(_Answer):
    type: Literal["choice"]
    choice: str = Field(description="The most probable option's key; among tied options the key that sorts first.")
    confidence: float = Field(description="System One's: the top probability rescaled from uniform.")
    probabilities: dict[str, float] = Field(description="The distribution over the options, by key.")


class ScoreAnswer(_Answer):
    type: Literal["score"]
    score: float = Field(description="The probability-weighted mean level.")
    confidence: float = Field(description="System One's: the concentration at the mode.")
    legend: dict[str, Any] = Field(description="Each level's index and description, as the request wrote it.")
    probabilities: dict[str, float] = Field(description="The distribution over the levels, by index.")


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class SystemOneResponse(BaseModel):
    model: str = Field(description="The model name the request sent, or the served model's name when it sent none.")
    answers: dict[str, Answer] = Field(description="By the request's question names.")
    usage: Usage
    adapter: str | None = Field(None, description="The adapter that answered, when the request named one.")
    thinking: dict[str, Any] | None = Field(
        None,
        description="When thinking was on for the request (`think`): `questions` that thought, `of`, `tokens`, `ms`, the "
        "`factor` and `budget`, and per question `thought` with its `tokens`, or why it did not.",
    )
    state_truncated: dict[str, Any] | None = Field(
        None,
        description="When the state was cut (`over_limit: truncate`): `kept_tokens`, `of_tokens`, `part` (`start`) and "
        "`limit`. No answer of a truncated state is certified.",
    )
    header_limited: dict[str, Any] | None = Field(
        None,
        description="When the header listed only some of the request's questions because the whole would not fit the "
        "context: `layout` (`header`, or `plain` when no line fit), `listed` of `of` lines, `header_tokens` and "
        "`of_header_tokens`, `row_limit`, `allowed_tokens` and `reason` (`context`, or `cap` for a --header-limit "
        "number). No answer of a certified task is certified.",
    )


class ModelEntry(BaseModel):
    name: str
    description: str
    release_date: str | None = None


class ModelList(BaseModel):
    models: list[ModelEntry]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    ok: bool
    engine: Any = Field(
        None,
        description="The engine's facts (vLLM version, GPU, mode, padding, quantization); the string `dead` once it has died.",
    )
    base: Any = Field(
        None,
        description="The base, the checkpoint and its revision, `engine_process`, the temperatures and the prompt.",
    )
    systemone: dict[str, Any] | None = Field(
        None,
        description="The System One route's settings: rendering, `noul_commit`, abstention, `orders`, tasks, temperatures.",
    )
    image_engine: Any = Field(
        None, description="The image engine's facts (including `max_images`), when `--image-model` is set."
    )


class DeadEngine(BaseModel):
    ok: Literal[False]
    engine: Literal["dead"]
    reason: str
    exit_code: int | None = None


class ErrorDetail(BaseModel):
    detail: str = Field(description="What is wrong.")


class ImageError(BaseModel):
    class Detail(BaseModel):
        error_type: Literal["api_usage_error"]
        message: str

    detail: Detail


class TaskExample(BaseModel):
    request: SystemOneRequest = Field(
        description="A /v1/systemone request with exactly one question (text only for POST /v1/tasks)."
    )
    answer: str | None = Field(description="The option key; null for an unanswerable example (abstention tasks).")


class CertifyBlock(BaseModel):
    alpha: float = Field(0.05, description="At most this share of the certified answers is wrong.")
    delta: float = Field(0.1, description="The promise fails with at most this probability.")
    min_examples: int | None = Field(
        None,
        description="Fewest held-out examples to accept; never fewer than alpha and delta need (59 at the defaults).",
    )
    holdout: float = Field(0.5, description="The share of the task's examples held out when `examples` is not given.")
    suspend_on_alert: bool = Field(
        True, description="Stop certifying while the traffic monitor's confidence alert is on."
    )
    examples: list[TaskExample] | None = Field(None, description="Labelled like the task's own; never fitted on.")


class TaskRegistration(BaseModel):
    id: str
    examples: list[TaskExample]
    certify: CertifyBlock | None = None


class CertifyRequest(BaseModel):
    examples: list[TaskExample]
    alpha: float | None = None
    delta: float | None = None
    min_examples: int | None = None
    suspend_on_alert: bool | None = None


class TaskImport(BaseModel):
    tasks: list[dict[str, Any]] = Field(description="Tasks as `GET /v1/tasks?full=1` returns them.")


class AbstentionRegistration(BaseModel):
    id: str
    option: dict[str, str] = Field(
        description='`{"key": "<the request\'s own abstain key>"}` or `{"append": "<text of an option the server adds>"}`.'
    )
    match: Literal["option_set", "option_keys", "imajev_extension"]
    examples: list[TaskExample]


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    format: str | None = None
    id: str | None = None
    key: str | None = None
    type: str | None = None
    options: list[str] | None = None
    fingerprint: str | None = None
    calibration: dict[str, Any] | None = None
    head: dict[str, Any] | None = None
    certificate: dict[str, Any] | None = None
    status: dict[str, Any] | None = None


MODELS = (
    SystemOneRequest, SystemOneResponse, NoulAnswer, ChoiceAnswer, ScoreAnswer, Usage, ModelList, ModelEntry, HealthResponse,
    DeadEngine, ErrorDetail, ImageError, TaskExample, CertifyBlock, TaskRegistration, CertifyRequest, TaskImport,
    AbstentionRegistration, TaskRecord,
)  # fmt: skip

RESPONSE_HEADERS = {
    "x-decisio-server-ms": "The server's time for the request, in milliseconds.",
    "x-decisio-route": "The route that served it: `text` or `image`.",
    "x-decisio-tasks": "The registered tasks applied, when any were.",
    "x-decisio-adapter": "The adapter that answered, when one did.",
    "x-decisio-think-ms": "The milliseconds spent on reasoning and the second reads, when thinking was on for the request.",
    "x-decisio-think-tokens": "The reasoning tokens generated, when thinking was on for the request.",
    "x-decisio-state-truncated": "`kept=N; of=M; part=start`, when the state was cut (`over_limit`).",
    "x-decisio-layout": "`header; listed=K; of=N; reason=context`, or `plain; ...`, when the header listed only some "
    "of the request's questions (`--header-limit`).",
    "x-decisio-stages": "Where the engine's time went, e.g. `prepare=1.2;warm=0.0;questions=52.1;readout=0.1;engine=53.6`.",
    "x-decisio-wait-behind": "`registration` when a boundary registration was running when the request arrived.",
}


def _ref(name: str) -> dict:
    return {"$ref": f"#/components/schemas/{name}"}


def _json(model: str, **extra) -> dict:
    return {"application/json": {"schema": _ref(model), **extra}}


def _resp(description: str, model: str | None = None, headers: bool = False) -> dict:
    out: dict[str, Any] = {"description": description}
    if model:
        out["content"] = _json(model)
    if headers:
        out["headers"] = {k: {"description": v, "schema": {"type": "string"}} for k, v in RESPONSE_HEADERS.items()}
    return out


# ---- the stub app -----------------------------------------------------------------------------------------------------


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [1]


class _DocEngine:
    """A stand-in engine that is never asked anything: building the app reads its attributes, nothing more. For documentation only."""

    tok = _Tok()
    multi_question = "sequential"
    pad_policy = "always"
    adapters: dict = {}

    def facts(self):
        return {"engine": "documentation stub"}


def build_app():
    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    engine = _DocEngine()
    return make_app(engine, SystemOne(engine, "decisio", task_store=TaskStore("documentation")))


# ---- OpenAPI ----------------------------------------------------------------------------------------------------------

SUMMARIES = {
    ("get", "/health"): "What the server is serving",
    ("post", "/v1/answer"): "The engine's own route (conformance gates and benchmarks)",
    ("get", "/v1/models"): "The served model, in System One's format",
    ("post", "/v1/systemone"): "TypeSafe's System One wire: answer typed questions about a state",
    ("get", "/v1/abstention/tasks"): "List per-task abstention thresholds",
    ("post", "/v1/abstention/tasks"): "Register an abstention threshold from labelled examples",
    (
        "post",
        "/v1/tasks",
    ): "Register a recurring question from labelled examples (calibration, intent head, certificate)",
    ("get", "/v1/tasks"): "List the registered tasks (`full=1` for the form `/v1/tasks/import` takes)",
    ("get", "/v1/tasks/{task_id}"): "One task with its status",
    ("delete", "/v1/tasks/{task_id}"): "Remove a task",
    ("post", "/v1/tasks/{task_id}/certify"): "Make a task's certificate again under the current serving stack",
    ("post", "/v1/tasks/import"): "Load tasks exported with `GET /v1/tasks?full=1`",
    ("get", "/metrics"): "Prometheus metrics (including the traffic monitor's)",
}

SYSTEMONE_PARAMETERS = [
    {"name": "x-decisio-route", "in": "header", "required": False, "schema": {"type": "string", "enum": ["text", "image"]},
     "description": "Choose the route explicitly; `image` sends a text request to the image engine (the paired check)."},
    {"name": "x-decisio-debug", "in": "header", "required": False, "schema": {"type": "string", "enum": ["readout", "hidden"]},
     "description": "Add the raw readout to the response; only on a server started with `--debug-readout` (verification, never production)."},
    {"name": "x-decisio-client", "in": "header", "required": False, "schema": {"type": "string"},
     "description": "Required, with any value, on a `multipart/form-data` request (a request a browser can send without a preflight is refused with 415)."},
]  # fmt: skip

MULTIPART = {
    "multipart/form-data": {
        "schema": {
            "type": "object",
            "required": ["request"],
            "properties": {
                "request": {
                    "description": "The JSON request, first.",
                    "contentMediaType": "application/json",
                    "contentSchema": _ref("SystemOneRequest"),
                },
                "image": {
                    "type": "string",
                    "format": "binary",
                    "description": "imajev's encoding: an image file (also `images`, `image[]`).",
                },
            },
            "additionalProperties": {
                "type": "string",
                "format": "binary",
                "description": "An image part named by the `part` of an asset (System One's encoding).",
            },
        }
    }
}


def openapi() -> dict:
    """The server's OpenAPI document with the wire's models."""
    d = build_app().openapi()
    d["info"] = {
        "title": "decisio",
        "version": decisio.__version__,
        "description": "The HTTP API of a decisio server: TypeSafe's System One wire on `POST /v1/systemone`, with routes for tasks, "
        "abstention, health and metrics. Generated from the server's code; `docs/api.md` is the prose.",
        "license": {"name": "Apache-2.0", "identifier": "Apache-2.0"},
    }
    d["servers"] = [{"url": "http://127.0.0.1:8000", "description": "The default host and port (`--host`, `--port`)"}]
    comp = d.setdefault("components", {}).setdefault("schemas", {})
    for m in MODELS:
        sch = m.model_json_schema(ref_template="#/components/schemas/{model}")
        comp.update(sch.pop("$defs", {}))
        comp[m.__name__] = sch
    if "Request" in comp:  # make_app's local model of POST /v1/answer: a name that says what it is
        comp["AnswerRequest"] = comp.pop("Request")
        comp["AnswerRequest"]["title"] = "AnswerRequest"
        d["paths"]["/v1/answer"]["post"]["requestBody"]["content"]["application/json"]["schema"] = _ref("AnswerRequest")
    for (method, path), summary in SUMMARIES.items():
        op = d["paths"].get(path, {}).get(method)
        if op is not None:
            op["summary"] = summary
    paths = d["paths"]
    so = paths["/v1/systemone"]["post"]
    so["description"] = (
        "A state and typed questions (`noul`, `choice`, `score`), answered with probabilities and System One's confidence. "
        "Images go in `assets` (placeholders in the texts), in `images` (imajev), as `data:image` URIs in the state, or in a multipart request; "
        "they are served by the image engine (a server started with `--image-model`)."
    )
    so["parameters"] = SYSTEMONE_PARAMETERS
    so["requestBody"] = {"required": True, "content": {**_json("SystemOneRequest"), **MULTIPART}}
    so["responses"] = {
        "200": _resp("The answers, by the request's question names.", "SystemOneResponse", headers=True),
        "400": {
            "description": "An unusable asset (`api_usage_error`) or a request the engine cannot answer (`detail` string).",
            "content": {"application/json": {"schema": {"oneOf": [_ref("ImageError"), _ref("ErrorDetail")]}}},
        },
        "404": _resp("An adapter that is not loaded.", "ErrorDetail"),
        "415": _resp("A POST that is not JSON (or a multipart request without `x-decisio-client`)."),
        "422": _resp(
            "A request that does not validate (for example a choice with more than 255 options).", "HTTPValidationError"
        ),
        "500": _resp(
            "The model's arithmetic produced a non-finite probability (not the request's fault).", "ErrorDetail"
        ),
        "503": _resp("The engine has died; the server is exiting.", "ErrorDetail"),
    }
    paths["/v1/models"]["get"]["responses"] = {"200": _resp("The served model.", "ModelList")}
    paths["/health"]["get"]["responses"] = {
        "200": _resp("What the server is serving.", "HealthResponse"),
        "503": _resp("The engine has died.", "DeadEngine"),
    }
    paths["/v1/tasks"]["post"]["requestBody"] = {"required": True, "content": _json("TaskRegistration")}
    paths["/v1/tasks"]["post"]["responses"] = {
        "200": _resp("The task's record: what was fitted, what is applied, the certificate.", "TaskRecord"),
        "422": _resp("Invalid.", "ErrorDetail"),
    }
    paths["/v1/tasks/{task_id}/certify"]["post"]["requestBody"] = {"required": True, "content": _json("CertifyRequest")}
    paths["/v1/tasks/{task_id}/certify"]["post"]["responses"] = {
        "200": _resp("The new certificate.", "TaskRecord"),
        "404": _resp("No such task.", "ErrorDetail"),
        "422": _resp("Invalid.", "ErrorDetail"),
    }
    paths["/v1/tasks/import"]["post"]["requestBody"] = {"required": True, "content": _json("TaskImport")}
    paths["/v1/abstention/tasks"]["post"]["requestBody"] = {
        "required": True,
        "content": _json("AbstentionRegistration"),
    }
    return d


# ---- flags ------------------------------------------------------------------------------------------------------------


def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace("|", "\\|").strip()


def _flag_row(a: argparse.Action) -> str | None:
    if not a.option_strings or a.dest == "help":
        return None
    flags = " / ".join(f"`{o}`" for o in a.option_strings)
    onoff = (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)
    if a.choices:
        values = ", ".join(f"`{c}`" for c in a.choices)
    elif isinstance(a, onoff):
        values = "on or off"
    elif a.metavar:
        values = f"`{a.metavar}`"
    elif a.type is not None:
        values = f"`{getattr(a.type, '__name__', str(a.type))}`"
    else:
        values = "text"
    if isinstance(a, argparse._AppendAction):
        values += ", repeatable"
    if isinstance(a, onoff):
        default = "-" if a.default is None else ("on" if a.default else "off")
    elif a.default in (None, argparse.SUPPRESS, []):
        default = "-"
    else:
        default = f"`{a.default}`"
    help_text = a.help or ""
    try:
        help_text = help_text % {**vars(a), "default": a.default, "prog": "decisio"}
    except (KeyError, ValueError, TypeError):
        pass
    return f"| {flags} | {default} | {values} | {_cell(help_text)} |"


def flags_markdown() -> str:
    from decisio.serve.vllm_engine import build_parser

    rows = [r for r in (_flag_row(a) for a in build_parser()._actions) if r]
    return (
        "<!-- SPDX-"
        "License-Identifier: Apache-2.0 -->\n"
        "<!-- SPDX-"
        "FileCopyrightText: Copyright contributors to the decisio project -->\n"
        "<!-- Generated by `python -m decisio.serve.reference_docs --write` from the server's argument parser. Do not edit. -->\n\n"
        "# Server flags\n\n"
        f"`python -m decisio.serve.vllm_engine`, decisio {decisio.__version__} <!-- x-release-please-version -->.\n"
        "A dash in the default column means the flag has no default of its own: it takes the base's value, or is off.\n"
        "`docs/cli.md` says which value each base takes.\n\n"
        "| Flag | Default | Values | What it does |\n| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n"
    )


# ---- files ------------------------------------------------------------------------------------------------------------


def render() -> dict[str, str]:
    return {"openapi.json": json.dumps(openapi(), indent=2, ensure_ascii=False) + "\n", "flags.md": flags_markdown()}


def same(name: str, text: str, path: Path) -> bool:
    """Whether the file is what the code gives: JSON by its content (release-please rewrites the version and may format the
    file its own way), Markdown by its text."""
    if not path.exists():
        return False
    have = path.read_text(encoding="utf-8")
    return json.loads(have) == json.loads(text) if name.endswith(".json") else have == text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="docs/reference")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    files = render()
    if a.write:
        out.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (out / name).write_text(text, encoding="utf-8")
            print(f"wrote {out / name}")
        return 0
    stale = [n for n, t in files.items() if not same(n, t, out / n)]
    for n in stale:
        print(
            f"{out / n} is not what the code gives: run `python -m decisio.serve.reference_docs --write`",
            file=sys.stderr,
        )
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
