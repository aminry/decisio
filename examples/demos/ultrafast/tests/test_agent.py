"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid System One"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("SYSTEMONE_BASE_URL", "http://stub")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("SYSTEMONE_BASE_URL", "http://stub")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid System One"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("SYSTEMONE_BASE_URL", "http://stub")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL", "stub-text")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_model_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL", "stub-text")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def test_choose_posts_to_the_configured_server_without_a_key_and_counts_options(monkeypatch):
    seen = {}

    def post(url, key, body):
        seen.update(url=url, key=key, body=body)
        ids = ["CLICK", "TYPE_TEXT", "WAIT", "DONE", "BLOCKED"]
        return {
            "model": "stub",
            "answers": {
                "operation": choice(ids, "CLICK"),
                "click_target": choice(["1", "2"], "2"),
                "type_text_target": choice(["1"], "1"),
            },
        }

    monkeypatch.setenv("SYSTEMONE_BASE_URL", "http://stub:9/")
    monkeypatch.delenv("SYSTEMONE_MODEL", raising=False)
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert seen["url"] == "http://stub:9/v1/systemone" and seen["key"] is None and "model" not in seen["body"]
    assert d["option_counts"]["operation"] == 5 and d["option_counts"]["click_target"] == 2
    assert d["latency_ms"] >= 0
    monkeypatch.setenv("SYSTEMONE_MODEL", "m")
    model.choose(page(), "Find a book", [])
    assert seen["body"]["model"] == "m"


def test_text_helper_needs_no_key_and_takes_extra_fields(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL", "stub-text")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://127.0.0.1:8200/v1")
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("TEXT_MODEL_EXTRA_JSON", '{"chat_template_kwargs": {"enable_thinking": false}}')
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    assert model.field_text({"goal": "Find a flight"})[0] == "Zurich"
    url, key, body = post.call_args.args
    assert url == "http://127.0.0.1:8200/v1/chat/completions" and key is None
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["model"] == "stub-text"


def test_string_descriptions_option_turns_json_descriptions_into_text(monkeypatch):
    seen = {}

    def post(url, key, body):
        seen["body"] = body
        ids = ["CLICK", "TYPE_TEXT", "WAIT", "DONE", "BLOCKED"]
        return {"model": "m", "answers": {"operation": choice(ids, "CLICK"), "click_target": choice(["1", "2"], "1"),
                                           "type_text_target": choice(["1"], "1")}}

    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.delenv("SYSTEMONE_STRING_DESCRIPTIONS", raising=False)
    model.choose(page(), "Find a book", [])
    assert isinstance(seen["body"]["questions"]["click_target"]["criteria"]["1"], dict)  # unchanged by default
    monkeypatch.setenv("SYSTEMONE_STRING_DESCRIPTIONS", "1")
    model.choose(page(), "Find a book", [])
    crit = seen["body"]["questions"]["click_target"]["criteria"]["1"]
    assert isinstance(crit, str) and json.loads(crit)["element"].startswith("[1]")


def test_skip_single_option_does_not_ask_a_one_option_head_and_answers_it_with_its_only_element(monkeypatch):
    seen = {}

    def post(url, key, body):
        seen["questions"] = body["questions"]
        ids = list(body["questions"]["operation"]["criteria"])
        return {"model": "stub", "answers": {"operation": choice(ids, "TYPE_TEXT")}}

    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.setenv("SYSTEMONE_SKIP_SINGLE_OPTION", "1")
    d = model.choose(page(), "Find a book", [])
    assert "type_text_target" not in seen["questions"]
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert d["target_probabilities"] == {"1": 1.0}
    assert d["skipped_single_option"] == ["type_text_target"]
    assert d["option_counts"]["type_text_target"] == 1


def test_done_submitted_variant_changes_only_dones_description(monkeypatch):
    seen = []

    def post(url, key, body):
        seen.append(body["questions"]["operation"]["criteria"])
        ids = list(body["questions"]["operation"]["criteria"])
        return {"model": "stub", "answers": {"operation": choice(ids, "DONE")}}

    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.delenv("ULTRAFAST_PROMPT_VARIANT", raising=False)
    model.choose(page(), "Find a book", [])
    monkeypatch.setenv("ULTRAFAST_PROMPT_VARIANT", "done-submitted")
    model.choose(page(), "Find a book", [])
    plain, variant = seen
    assert plain["DONE"] == "Every requirement is visibly satisfied."
    assert "not been submitted" in variant["DONE"]
    assert {k: v for k, v in plain.items() if k != "DONE"} == {k: v for k, v in variant.items() if k != "DONE"}
