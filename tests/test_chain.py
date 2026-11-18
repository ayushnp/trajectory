"""M3: sessions, steps, turns. Conversations are built the way real agents build them."""
import pytest

from trajectory.chain import Chainer, digest_request

S = 1_000_000_000  # 1 s in ns


# ---------------------------------------------------------------- conversation builders
def a_user(text):
    return {"role": "user", "content": text}


def a_assistant_tool(tid, name="Read", args=None, thinking=False):
    blocks = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": "hmm", "signature": "sig"})
    blocks.append({"type": "tool_use", "id": tid, "name": name, "input": args or {"p": "a"}})
    return {"role": "assistant", "content": blocks}


def a_result(tid, text="ok"):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}


def a_assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def rec(msgs, t=0, key="k", fmt="anthropic", status=200, call_id=None, model="m1"):
    digests, human, opaque = digest_request(fmt, {"input" if fmt == "openai_responses" else "messages": msgs})
    return {"client": {"key_hash": key},
            "request": {"prefix_digests": digests, "human_indexes": human, "opaque_reasoning": opaque,
                        "format": fmt, "model_requested": model},
            "timing": {"t_received": t * S}, "response": {"status": status},
            "outcome": {"retry_index": 0}, "call_id": call_id}


def chain(chainer, *recs):
    for r in recs:
        chainer.assign(r)
    return recs


def ids(rs):
    return [(r["session_id"], r["step_index"], r["turn_index"]) for r in rs]


# ---------------------------------------------------------------- the five-step conversation
def test_five_step_conversation_is_one_session_with_increasing_steps():
    c = Chainer()
    m = [a_user("fix the bug")]
    rs = [rec(m, 0)]
    for i in range(4):
        m = m + [a_assistant_tool(f"t{i}"), a_result(f"t{i}")]
        rs.append(rec(m, 10 * (i + 1)))
    chain(c, *rs)
    assert len({r["session_id"] for r in rs}) == 1
    assert [r["step_index"] for r in rs] == [0, 1, 2, 3, 4]
    assert [r["turn_index"] for r in rs] == [0, 0, 0, 0, 0]       # tool results are not human turns
    assert not any(r.get("compaction_suspected") for r in rs)


def test_new_human_message_starts_a_new_turn_but_same_session():
    c = Chainer()
    m1 = [a_user("first question")]
    m2 = m1 + [a_assistant("answer"), a_user("second question")]     # human message added -> turn 1
    m3 = m2 + [a_assistant_tool("t1"), a_result("t1")]               # tool result only -> still turn 1
    m4 = m3 + [a_assistant("done")]
    r1, r2, r3, r4 = chain(c, rec(m1, 0), rec(m2, 5), rec(m3, 10), rec(m4, 12))
    assert r1["session_id"] == r2["session_id"] == r3["session_id"] == r4["session_id"]
    assert [r["step_index"] for r in (r1, r2, r3, r4)] == [0, 1, 2, 3]
    assert [r["turn_index"] for r in (r1, r2, r3, r4)] == [0, 1, 1, 1]


def test_tool_result_user_message_is_not_a_human_turn():
    d, human, _ = digest_request("anthropic", {"messages": [a_user("hi"), a_assistant_tool("t"), a_result("t")]})
    assert human == [0]


def test_tool_result_with_extra_text_is_still_not_human():
    mixed = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"},
                                         {"type": "text", "text": "system reminder"}]}
    _, human, _ = digest_request("anthropic", {"messages": [a_user("hi"), a_assistant_tool("t"), mixed]})
    assert human == [0]


# ---------------------------------------------------------------- parallel chains
def test_two_parallel_chains_with_different_starts_stay_separate():
    c = Chainer()
    a1, b1 = [a_user("task A")], [a_user("task B")]
    a2, b2 = a1 + [a_assistant_tool("a")], b1 + [a_assistant_tool("b")]
    rs = chain(c, rec(a1, 0), rec(b1, 1), rec(a2, 5), rec(b2, 6))
    assert rs[0]["session_id"] == rs[2]["session_id"] != rs[1]["session_id"] == rs[3]["session_id"]
    assert [r["step_index"] for r in rs] == [0, 0, 1, 1]


def test_parallel_chains_from_different_clients_never_mix_even_with_identical_prompts():
    c = Chainer()
    m = [a_user("same prompt")]
    r1, r2 = chain(c, rec(m, 0, key="alice"), rec(m, 1, key="bob"))
    assert r1["session_id"] != r2["session_id"]


def test_same_client_same_first_message_diverging_after_forks_into_two_sessions():
    """A subagent/fork shares the first message with its parent, then goes its own way."""
    c = Chainer()
    root = [a_user("shared start")]
    parent2 = root + [a_assistant_tool("p1")]
    child2 = root + [a_assistant_tool("c1", name="Grep")]            # diverges at message 1
    r0, rp, rc = chain(c, rec(root, 0), rec(parent2, 5), rec(child2, 6))
    assert r0["session_id"] == rp["session_id"]
    assert rc["session_id"] != rp["session_id"]
    assert rc["continued_from_session"] == rp["session_id"] or rc["continued_from_session"] == r0["session_id"]


# ---------------------------------------------------------------- compaction
def test_shrink_is_flagged_as_compaction_and_links_back():
    c = Chainer()
    m = [a_user("start")]
    long = [m[0]]
    for i in range(6):
        long += [a_assistant_tool(f"t{i}"), a_result(f"t{i}")]
    r_long = rec(long, 0)
    compacted = [a_user("start"), a_assistant("summary of everything"), a_user("continue")]
    r_comp = rec(compacted, 10)
    chain(c, rec([long[0]], -5), r_long, r_comp)
    assert r_comp["session_id"] != r_long["session_id"]
    assert r_comp["compaction_suspected"] is True
    assert r_comp["continued_from_session"] == r_long["session_id"]
    assert r_comp["step_index"] == 0


def test_mid_history_edit_is_treated_as_divergence():
    c = Chainer()
    base = [a_user("q"), a_assistant("a1"), a_user("q2")]
    edited = [a_user("q"), a_assistant("CHANGED"), a_user("q2"), a_assistant("a3")]
    r1, r2 = chain(c, rec(base, 0), rec(edited, 5))
    assert r2["session_id"] != r1["session_id"] and r2["compaction_suspected"]


def test_unrelated_conversation_is_a_clean_new_session_without_compaction_flag():
    c = Chainer()
    r1, r2 = chain(c, rec([a_user("topic one")], 0), rec([a_user("totally different")], 5))
    assert r1["session_id"] != r2["session_id"]
    assert not r2.get("compaction_suspected") and "continued_from_session" not in r2


# ---------------------------------------------------------------- retries, errors, decay
def test_identical_request_is_a_retry_not_a_new_step():
    c = Chainer()
    m = [a_user("hi")]
    r1, r2, r3 = chain(c, rec(m, 0), rec(m, 2), rec(m, 4))
    assert r1["session_id"] == r2["session_id"] == r3["session_id"]
    assert r2["step_index"] == r3["step_index"] == 0
    assert (r2["outcome"]["retry_index"], r3["outcome"]["retry_index"]) == (1, 2)


def test_failed_call_does_not_become_the_chain_head():
    c = Chainer()
    m1 = [a_user("hi")]
    m2 = m1 + [a_assistant_tool("t"), a_result("t")]
    r1, rfail, r3 = chain(c, rec(m1, 0), rec(m2, 5, status=500), rec(m2, 8))
    assert rfail["session_id"] == r1["session_id"] and rfail["step_index"] == 1
    assert r3["session_id"] == r1["session_id"] and r3["step_index"] == 1   # the retry takes the slot


def test_candidates_decay_after_ttl():
    c = Chainer(ttl_s=60)
    m1 = [a_user("hi")]
    m2 = m1 + [a_assistant("a"), a_user("later")]
    r1, r2 = chain(c, rec(m1, 0), rec(m2, 3600))
    assert r1["session_id"] != r2["session_id"]


def test_max_steps_bounds_a_session_but_keeps_the_link():
    c = Chainer(max_steps=3)
    m, rs = [a_user("go")], []
    rs.append(rec(m, 0))
    for i in range(4):
        m = m + [a_assistant_tool(f"t{i}"), a_result(f"t{i}")]
        rs.append(rec(m, i + 1))
    chain(c, *rs)
    assert rs[3]["session_id"] != rs[2]["session_id"]
    assert rs[3]["continued_from_session"] == rs[2]["session_id"]
    assert not rs[3].get("compaction_suspected")


def test_head_memory_is_bounded():
    c = Chainer(max_heads_per_client=10)
    for i in range(500):
        c.assign(rec([a_user(f"unrelated {i}")], i))
    assert max(len(v) for v in c._heads.values()) <= 10 and len(c._heads) == 1


# ---------------------------------------------------------------- reasoning is opaque
def test_dropped_thinking_blocks_do_not_break_continuation_but_are_flagged():
    c = Chainer()
    step1 = [a_user("go")]
    with_thinking = step1 + [a_assistant_tool("t", thinking=True), a_result("t")]
    without = step1 + [a_assistant_tool("t", thinking=False), a_result("t")]
    r1, r2, r3 = chain(c, rec(step1, 0), rec(with_thinking, 5), rec(without, 8))
    assert r1["session_id"] == r2["session_id"]
    assert r2["request"]["opaque_reasoning"] is True and r1["request"]["opaque_reasoning"] is False
    assert r3["session_id"] == r1["session_id"]       # same meaning for chaining purposes


def test_openai_responses_reasoning_items_are_skipped_in_digest():
    base = [{"role": "user", "content": "hi"}]
    a = digest_request("openai_responses", {"input": base})
    b = digest_request("openai_responses", {"input": base[:1] + [{"type": "reasoning", "encrypted_content": "zzz"}]})
    assert a[0] == b[0] and b[2] is True


# ---------------------------------------------------------------- format coverage
def test_openai_chat_conversation_chains_and_detects_human_turns():
    c = Chainer()
    m1 = [{"role": "user", "content": "list files"}]
    m2 = m1 + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": '{"d":"."}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a b c"}]
    m3 = m2 + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "thanks, now delete"}]
    r1, r2, r3 = chain(c, rec(m1, 0, fmt="openai_chat"), rec(m2, 3, fmt="openai_chat"), rec(m3, 6, fmt="openai_chat"))
    assert len({r["session_id"] for r in (r1, r2, r3)}) == 1
    assert [r["turn_index"] for r in (r1, r2, r3)] == [0, 0, 1]


def test_openai_responses_string_input_and_item_lists_chain():
    c = Chainer()
    first = digest_request("openai_responses", {"input": "hello"})
    second = digest_request("openai_responses", {"input": [
        {"role": "user", "content": "hello"},
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": "x"}]})
    assert second[0][:1] == first[0]


def test_json_tool_arguments_normalize_regardless_of_key_order_or_string_form():
    a = digest_request("openai_chat", {"messages": [{"role": "assistant", "tool_calls": [
        {"id": "1", "function": {"name": "f", "arguments": '{"b":1,"a":2}'}}]}]})
    b = digest_request("openai_chat", {"messages": [{"role": "assistant", "tool_calls": [
        {"id": "1", "function": {"name": "f", "arguments": '{ "a": 2, "b": 1 }'}}]}]})
    assert a[0] == b[0]


# ---------------------------------------------------------------- robustness
@pytest.mark.parametrize("rq", [{}, {"prefix_digests": []}])
def test_records_without_digests_are_left_alone(rq):
    c = Chainer()
    r = {"client": {"key_hash": "k"}, "request": rq, "timing": {"t_received": 0},
         "response": {"status": 200}, "outcome": {}}
    c.assign(r)
    assert "session_id" not in r


def test_garbage_message_shapes_do_not_raise():
    for msgs in ([None, 1, "x", [], {"role": 5}], [{"role": "user", "content": [None, 3, {"type": 7}]}]):
        for fmt in ("anthropic", "openai_chat"):
            digest_request(fmt, {"messages": msgs})
    digest_request("openai_responses", {"input": [None, 1, {"type": "message", "content": [None]}]})


def test_same_prompt_to_a_different_model_is_not_a_retry_or_continuation():
    c = Chainer()
    m = [a_user("Write two sentences about rivers.")]
    r1, r2 = chain(c, rec(m, 0, model="claude-haiku-5-5"), rec(m, 1, model="claude-sonnet-5-5"))
    assert r1["session_id"] != r2["session_id"]
    assert r2["outcome"]["retry_index"] == 0 and r2["step_index"] == 0


def test_same_prompt_on_a_different_endpoint_is_a_separate_session():
    c = Chainer()
    text = "hello there"
    a = rec([{"role": "user", "content": text}], 0, fmt="openai_chat", model="gpt-6-luna")
    b = rec([{"role": "user", "content": text}], 1, fmt="openai_responses", model="gpt-6-luna")
    chain(c, a, b)
    assert a["session_id"] != b["session_id"]


def test_true_retry_on_the_same_model_is_still_a_retry():
    c = Chainer()
    m = [a_user("hi")]
    r1, r2 = chain(c, rec(m, 0, model="x"), rec(m, 1, model="x"))
    assert r1["session_id"] == r2["session_id"] and r2["outcome"]["retry_index"] == 1


# ---------------------------------------------------------------- compaction summaries
SUMMARY = "This session is being continued from a previous conversation that ran out of context."


def rec_summary(msgs, t=0, key="k", model="m1"):
    r = rec(msgs, t, key=key, model=model)
    from trajectory.chain import is_compaction_summary
    r["request"]["continues_prior_context"] = is_compaction_summary(
        "anthropic", {"messages": msgs})
    return r


def test_summary_first_message_links_to_latest_session_even_though_no_digest_matches():
    c = Chainer()
    old = [a_user("start"), a_assistant_tool("t0"), a_result("t0")]
    r_old = rec(old, 0)
    compacted = [a_user(SUMMARY + " ... summary body"), a_user("continue")]
    r_new = rec_summary(compacted, 10)
    chain(c, r_old, r_new)
    assert r_new["session_id"] != r_old["session_id"]
    assert r_new["continued_from_session"] == r_old["session_id"]
    assert r_new["compaction_suspected"] is True


def test_summary_does_not_link_across_models():
    c = Chainer()
    r_old = rec([a_user("start")], 0, model="m1")
    r_new = rec_summary([a_user(SUMMARY + " body")], 10, model="m2")
    chain(c, r_old, r_new)
    assert "continued_from_session" not in r_new


def test_summary_with_no_live_session_starts_a_clean_session():
    c = Chainer()
    r_new = rec_summary([a_user(SUMMARY + " body")], 10)
    chain(c, r_new)
    assert "continued_from_session" not in r_new
    assert r_new["step_index"] == 0


def test_ordinary_first_message_is_not_a_summary():
    from trajectory.chain import is_compaction_summary
    assert is_compaction_summary("anthropic", {"messages": [a_user("hello")]}) is False
    assert is_compaction_summary("openai_chat", {"messages": [a_user(SUMMARY)]}) is False
