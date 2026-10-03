"""A System One server makes choices; a small OpenAI-compatible text model writes field values.

Changed from upstream (see ../ATTRIBUTION.md): the choices come from `<SYSTEMONE_BASE_URL>/v1/systemone` on a server you
run, with no key; the text helper is any OpenAI-compatible server (`TEXT_MODEL_BASE_URL`), by default a local one, and its
key is optional. The request bodies, the questions and the validation are unchanged. Every result also records how many
options each question asked over (`option_counts`)."""

import json
import math
import os
import time

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


def post_json(url, key, body):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers=headers)
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid System One response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def stringify_descriptions(questions):
    """For a server that takes an option's description only as text (Ollama's route does): each JSON description becomes
    its compact JSON string, so the model reads the same content. Opt in with SYSTEMONE_STRING_DESCRIPTIONS=1."""
    for question in questions.values():
        question["criteria"] = {
            key: value if value is None or isinstance(value, str) else json.dumps(value, separators=(",", ":"))
            for key, value in question["criteria"].items()
        }
    return questions


def choose(state, goal, history):
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    done = "Every requirement is visibly satisfied."
    # Opt-in wording variant for testing a fix against the oracle (added for this repository; the default is unchanged):
    # DONE's description restates the instructions' own rule that a typed search is applied only once submitted.
    if os.environ.get("ULTRAFAST_PROMPT_VARIANT") == "done-submitted":
        done = "Every requirement is visibly satisfied; a typed search that has not been submitted is not applied yet."
    operations.update(DONE=done, BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    option_counts = {name: len(q["criteria"]) for name, q in questions.items()}
    if os.environ.get("SYSTEMONE_STRING_DESCRIPTIONS") == "1":
        stringify_descriptions(questions)
    # Opt in with SYSTEMONE_SKIP_SINGLE_OPTION=1 for a server that refuses a question with one option (the Ollama route
    # takes 2 to 26): that head is not asked, and its only element is its answer, with probability 1.
    sole = {}
    if os.environ.get("SYSTEMONE_SKIP_SINGLE_OPTION") == "1":
        for name in [n for n, q in questions.items() if n != "operation" and len(q["criteria"]) == 1]:
            index = next(iter(questions.pop(name)["criteria"]))
            sole[name] = {"type": "choice", "choice": index, "probabilities": {index: 1.0}, "confidence": 1.0}
    body = {
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    if os.environ.get("SYSTEMONE_MODEL"):
        body["model"] = os.environ["SYSTEMONE_MODEL"]
    started = time.perf_counter()
    result = post_json(os.environ.get("SYSTEMONE_BASE_URL", "http://127.0.0.1:8100").rstrip("/") + "/v1/systemone", None, body)
    result["answers"] = {**result["answers"], **sole}
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "option_counts": option_counts,
        "skipped_single_option": sorted(sole),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")  # optional: a local server needs none
    base = os.environ.get("TEXT_MODEL_BASE_URL", "http://127.0.0.1:8200/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL")
    if not model:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL (the text helper's model name); no text is hardcoded or guessed.")
    # Extra request fields for the helper, as JSON: a local Qwen3 server wants {"chat_template_kwargs": {"enable_thinking": false}}.
    reasoning = json.loads(os.environ.get("TEXT_MODEL_EXTRA_JSON") or "{}")
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
