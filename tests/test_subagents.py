"""M5: subagent attachment, confidence labels and the unattached bucket."""
import json

from tux_trace.subagents import MIN_ECHO_CHARS, SubagentAttacher, first_human_text, normalize

S = 1_000_000_000
PROMPT = "Read the file C:\\Users\\me\\proj\\a.txt and report its exact contents back, verbatim."


def parent(call_id="p1", session="sP", t_received=0, t_end=2, tools=None, key="k"):
    tools = tools if tools is not None else [
        {"id": "toolu_agent", "name": "Agent", "args": json.dumps({"description": "read", "prompt": PROMPT})}]
    return {"call_id": call_id, "session_id": session, "step_index": 0, "client": {"key_hash": key},
            "timing": {"t_received": t_received * S, "t_end": t_end * S},
            "request": {"tool_results": []}, "outcome": {"retry_index": 0},
            "response": {"tool_calls": tools}}


def child(call_id="c1", session="sC", t_received=3, step=0, key="k", retry=0):
    return {"call_id": call_id, "session_id": session, "step_index": step, "client": {"key_hash": key},
            "timing": {"t_received": t_received * S, "t_end": (t_received + 1) * S},
            "request": {"tool_results": []}, "outcome": {"retry_index": retry},
            "response": {"tool_calls": []}}


def feed(a, rec, first_text=None):
    a.note_results(rec)
    a.assess(rec, first_text)
    a.register(rec)
    return rec


# ---------------------------------------------------------------- confidence levels
def test_echo_and_overlap_attach_at_high_confidence():
    a = SubagentAttacher()
    feed(a, parent())
    c = feed(a, child(), first_text=PROMPT)
    assert c["parent_call_id"] == "p1"
    assert c["subagent_edge"] == {"parent_call_id": "p1", "tool_call_id": "toolu_agent", "confidence": "high",
                                  "signals": ["echo", "overlap"], "score": 0.85}
    assert "unattached" not in c or c["unattached"] is False


def test_overlap_only_attaches_at_medium():
    a = SubagentAttacher()
    feed(a, parent())
    c = feed(a, child(), first_text="a completely different instruction to follow")
    assert c["parent_call_id"] == "p1"
    assert c["subagent_edge"]["confidence"] == "medium" and c["subagent_edge"]["signals"] == ["overlap"]


def test_echo_only_is_low_and_never_attached():
    """The child arrives after the parent call was answered: no overlap, only a text match."""
    a = SubagentAttacher()
    p = feed(a, parent())
    answer = child("r", "sP", t_received=4, step=1)
    answer["request"]["tool_results"] = [{"tool_call_id": "toolu_agent", "repeat": False}]
    feed(a, answer)
    c = feed(a, child("late", "sL", t_received=10), first_text=PROMPT)
    assert c.get("parent_call_id") is None
    assert c["subagent_edge"]["confidence"] == "low" and c["subagent_edge"]["attached"] is False
    assert not c.get("unattached")


def test_no_signal_means_no_edge_at_all():
    a = SubagentAttacher()
    feed(a, parent())
    answered = child("r", "sP", t_received=4, step=1)
    answered["request"]["tool_results"] = [{"tool_call_id": "toolu_agent", "repeat": False}]
    feed(a, answered)
    c = feed(a, child("x", "sX", t_received=20), first_text="something unrelated entirely here")
    assert "subagent_edge" not in c and c.get("parent_call_id") is None and not c.get("unattached")


# ---------------------------------------------------------------- the unattached bucket
def test_helper_attaches_to_the_call_that_issued_its_tools_even_when_the_tool_is_unknown():
    """Agent + Read issued together, child prompt matches neither: the parent CALL is still certain."""
    a = SubagentAttacher()
    feed(a, parent(tools=[{"id": "t1", "name": "Agent", "args": "{}"}, {"id": "t2", "name": "Read", "args": "{}"}]))
    c = feed(a, child(), first_text="a prompt that appears in neither call")
    assert c["parent_call_id"] == "p1" and not c.get("unattached")
    assert c["subagent_edge"]["tool_call_id"] is None and c["subagent_edge"]["confidence"] == "medium"
    assert a.attached == 1 and a.unattached == 0 and a.unattached_rate == 0.0


def test_echo_resolves_the_ambiguity_between_agent_and_read():
    """Same shape, but the child's prompt is in the Agent call: attach to that one, not the Read."""
    a = SubagentAttacher()
    feed(a, parent(tools=[{"id": "t1", "name": "Agent", "args": json.dumps({"prompt": PROMPT})},
                          {"id": "t2", "name": "Read", "args": json.dumps({"file_path": "x"})}]))
    c = feed(a, child(), first_text=PROMPT)
    assert c["subagent_edge"]["confidence"] == "high" and c["parent_call_id"] == "p1"
    assert not c.get("unattached")


def test_other_clients_pending_calls_are_not_candidates_at_all():
    a = SubagentAttacher()
    feed(a, parent(key="alice"))
    c = feed(a, child(key="bob"), first_text="independent work by someone else")
    assert c.get("parent_call_id") is None and not c.get("unattached")
    assert a.child_candidates == 0


def test_overlap_with_exactly_one_pending_call_is_medium():
    a = SubagentAttacher()
    feed(a, parent())
    c = feed(a, child(), first_text="unrelated words that match nothing in the call")
    assert c["subagent_edge"]["confidence"] == "medium" and not c.get("unattached")


def test_unattached_rate_counts_candidates_that_found_no_parent():
    a = SubagentAttacher()
    a.child_candidates, a.unattached = 4, 1
    assert a.unattached_rate == 0.25
    assert SubagentAttacher().unattached_rate is None            # no candidates: undefined, not 0


# ---------------------------------------------------------------- timing windows
def test_child_before_parent_response_ended_attaches_only_if_the_call_had_appeared():
    """Without a per-call arrival time we fall back to the response end, so this must not attach."""
    a = SubagentAttacher()
    feed(a, parent(t_end=10))
    c = feed(a, child(t_received=5), first_text=PROMPT)
    assert c.get("parent_call_id") is None


def test_pending_window_closes_when_result_is_delivered():
    a = SubagentAttacher()
    feed(a, parent())
    r = child("r", "sP", t_received=4, step=1)
    r["request"]["tool_results"] = [{"tool_call_id": "toolu_agent", "repeat": False}]
    feed(a, r)
    c = feed(a, child("late", "sL", t_received=5), first_text=PROMPT)
    assert c.get("parent_call_id") is None


def test_pending_window_expires_after_ttl():
    a = SubagentAttacher(ttl_s=60)
    feed(a, parent())
    c = feed(a, child(t_received=5000), first_text=PROMPT)
    assert c.get("parent_call_id") is None


# ---------------------------------------------------------------- who can be a child
def test_only_chain_roots_can_be_children():
    a = SubagentAttacher()
    feed(a, parent())
    c = feed(a, child(step=3), first_text=PROMPT)
    assert c.get("parent_call_id") is None and "subagent_edge" not in c


def test_retries_are_not_children():
    a = SubagentAttacher()
    feed(a, parent())
    c = feed(a, child(retry=1), first_text=PROMPT)
    assert "subagent_edge" not in c


def test_a_session_never_attaches_to_its_own_tool_call():
    a = SubagentAttacher()
    feed(a, parent(session="same"))
    c = feed(a, child(session="same"), first_text=PROMPT)
    assert c.get("parent_call_id") is None


def test_other_clients_calls_are_never_parents():
    a = SubagentAttacher()
    feed(a, parent(key="alice"))
    c = feed(a, child(key="bob"), first_text=PROMPT)
    assert c.get("parent_call_id") is None


def test_two_pending_calls_from_different_sessions_without_echo_are_not_guessed():
    a = SubagentAttacher()
    feed(a, parent("p_old", "s1", t_end=2, tools=[{"id": "t_old", "name": "Agent", "args": "{}"}]))
    feed(a, parent("p_new", "s2", t_end=4, tools=[{"id": "t_new", "name": "Agent", "args": "{}"}]))
    c = feed(a, child(t_received=6), first_text="nothing in common with either call")
    assert c.get("parent_call_id") is None and c["unattached"] is True


def test_echo_beats_recency():
    a = SubagentAttacher()
    feed(a, parent("p_match", "s1", t_end=2))
    feed(a, parent("p_recent", "s2", t_end=4, tools=[{"id": "t2", "name": "Agent", "args": "{}"}]))
    c = feed(a, child(t_received=6), first_text=PROMPT)
    assert c["parent_call_id"] == "p_match" and c["subagent_edge"]["confidence"] == "high"


# ---------------------------------------------------------------- matching details
def test_short_prompts_are_not_treated_as_an_echo():
    assert len("ok") < MIN_ECHO_CHARS
    a = SubagentAttacher()
    feed(a, parent(tools=[{"id": "t", "name": "Agent", "args": json.dumps({"prompt": "ok go ahead ok"})}]))
    c = feed(a, child(), first_text="ok")
    assert c["subagent_edge"]["confidence"] == "medium"


def test_echo_survives_json_escaping_whitespace_and_case():
    args = json.dumps({"prompt": "Read the file  C:\\Users\\me\\a.txt\nand REPORT it"})
    assert normalize("read the file c:\\users\\me\\a.txt and report it") in normalize(args)
    a = SubagentAttacher()
    feed(a, parent(tools=[{"id": "t", "name": "Agent", "args": args}]))
    c = feed(a, child(), first_text="Read the file C:\\Users\\me\\a.txt\n   and report it")
    assert c["subagent_edge"]["confidence"] == "high"


def test_only_first_200_chars_of_child_prompt_are_compared():
    long_prompt = "x" * 300
    assert len(normalize(long_prompt)) == 200


# ---------------------------------------------------------------- first human text
def test_first_human_text_skips_tool_results_and_handles_formats():
    body = {"messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
        {"role": "user", "content": [{"type": "text", "text": "the real prompt"}]}]}
    assert first_human_text("anthropic", body) == "the real prompt"
    assert first_human_text("openai_chat", {"messages": [{"role": "user", "content": "hello there"}]}) == "hello there"
    assert first_human_text("openai_responses", {"input": "plain string"}) == "plain string"
    assert first_human_text("unknown", body) is None
    assert first_human_text("anthropic", {"messages": [None, 3]}) is None


# ---------------------------------------------------------------- privacy
def test_prompt_text_never_reaches_the_record():
    a = SubagentAttacher()
    feed(a, parent())
    c = feed(a, child(), first_text=PROMPT)
    assert "report its exact contents" not in json.dumps(c)


# ---------------------------------------------------------------- end to end through the real writer
def test_end_to_end_through_proxy(proxy_factory):
    import time

    import httpx

    def ev(t, o):
        return f"event: {t}\ndata: {json.dumps(o)}\n\n".encode()

    def tool_sse(name, tid, args):
        return [ev("message_start", {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 1}}}),
                ev("content_block_start", {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "tool_use", "id": tid, "name": name, "input": args}}),
                ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}}),
                ev("message_stop", {"type": "message_stop"})]

    proxy, _, _ = proxy_factory(tool_sse("Agent", "toolu_A", {"prompt": PROMPT}))
    h = {"x-api-key": "k"}
    httpx.post(proxy + "/v1/messages", headers=h, content=json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "delegate this"}]}))
    time.sleep(0.3)                               # parent has finished, its tool call is pending
    httpx.post(proxy + "/v1/messages", headers=h, content=json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": PROMPT}]}))   # the subagent's call
    end, rows = time.time() + 5, []
    while time.time() < end and len(rows) < 2:
        rows = httpx.get(proxy + "/_trace/api/calls?limit=10").json()["calls"]
        time.sleep(0.1)
    full = sorted((httpx.get(proxy + f"/_trace/api/calls/{r['call_id']}").json() for r in rows),
                  key=lambda r: r["timing"]["received_at"])
    par, kid = full
    assert kid["parent_call_id"] == par["call_id"]
    assert kid["subagent_edge"]["confidence"] == "high"
    assert kid["session_id"] != par["session_id"]
    assert PROMPT not in json.dumps(kid)
    stats = httpx.get(proxy + "/stats").json()
    assert stats["subagents_attached"] == 1 and stats["unattached_rate"] == 0.0


def test_child_starting_before_parent_stream_ends_still_attaches():
    """Real Claude Code behaviour: the subagent launches as soon as the Agent call appears,
    ~90 ms before the parent's response finishes streaming."""
    a = SubagentAttacher()
    tools = [{"id": "toolu_agent", "name": "Agent", "t_start": 1 * S,
              "args": json.dumps({"prompt": PROMPT})}]
    feed(a, parent(t_end=2, tools=tools))                         # response ends at t=2 s
    c = feed(a, child(t_received=1.5), first_text=PROMPT)         # child arrives at t=1.5 s
    assert c["parent_call_id"] == "p1" and c["subagent_edge"]["confidence"] == "high"


def test_child_before_the_call_even_appeared_does_not_attach():
    a = SubagentAttacher()
    tools = [{"id": "toolu_agent", "name": "Agent", "t_start": 5 * S, "args": json.dumps({"prompt": PROMPT})}]
    feed(a, parent(t_end=6, tools=tools))
    c = feed(a, child(t_received=3), first_text=PROMPT)
    assert c.get("parent_call_id") is None


# ---------------------------------------------------------------- parallel and nested subagents
SHARED = "You are a code reviewer for the payments service. Follow the checklist in docs/review.md carefully. "


def agent(tid, prompt):
    return {"id": tid, "name": "Agent", "args": json.dumps({"prompt": prompt})}


def test_parallel_children_with_distinct_prompts_each_get_their_own_parent():
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", "Summarise the billing module in detail please"),
                          agent("t2", "List every TODO in the refunds module please")]))
    c1 = feed(a, child("c1", "s1", t_received=3), first_text="Summarise the billing module in detail please")
    c2 = feed(a, child("c2", "s2", t_received=3), first_text="List every TODO in the refunds module please")
    assert c1["subagent_edge"]["confidence"] == c2["subagent_edge"]["confidence"] == "high"
    assert a.attached == 2 and a.unattached == 0


def test_parallel_children_sharing_a_200_char_prefix_are_told_apart_by_the_rest_of_the_prompt():
    p1, p2 = SHARED * 3 + "Review file alpha.py.", SHARED * 3 + "Review file beta.py."
    assert normalize(p1) == normalize(p2)                      # the 200-char needle cannot separate them
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", p1), agent("t2", p2)]))
    c1 = feed(a, child("c1", "s1", t_received=3), first_text=p1)
    c2 = feed(a, child("c2", "s2", t_received=3), first_text=p2)
    assert c1["subagent_edge"]["parent_call_id"] == "p1" and c1["subagent_edge"]["confidence"] == "high"
    assert c2["subagent_edge"]["parent_call_id"] == "p1" and c2["subagent_edge"]["confidence"] == "high"
    # and they pair with different tool calls, not the same one
    assert c1["subagent_edge"]["tool_call_id"] == "t1" and c2["subagent_edge"]["tool_call_id"] == "t2"


def test_identical_sibling_prompts_all_attach_to_the_same_parent_call():
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", SHARED * 3), agent("t2", SHARED * 3), agent("t3", SHARED * 3)]))
    kids = [feed(a, child(f"c{i}", f"s{i}", t_received=3), first_text=SHARED * 3) for i in range(3)]
    assert [k["parent_call_id"] for k in kids] == ["p1"] * 3
    assert all(k["subagent_edge"]["confidence"] == "high" and not k.get("unattached") for k in kids)
    assert a.unattached == 0


def test_a_call_already_claimed_at_high_confidence_is_eliminated_for_the_next_child():
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", "Summarise the billing module in detail please"),
                          {"id": "t2", "name": "Agent", "args": "{}"}]))        # prompt not recoverable
    c1 = feed(a, child("c1", "s1", t_received=3), first_text="Summarise the billing module in detail please")
    c2 = feed(a, child("c2", "s2", t_received=3), first_text="a prompt that is nowhere in the arguments")
    assert c1["subagent_edge"]["confidence"] == "high"
    assert c2["parent_call_id"] == "p1" and c2["subagent_edge"]["confidence"] == "medium"
    assert c2["subagent_edge"]["signals"] == ["overlap", "elimination"]
    assert c2["subagent_edge"]["tool_call_id"] == "t2"


def test_a_second_helper_from_the_same_single_call_attaches_to_it_with_the_tool_left_blank():
    a = SubagentAttacher()
    feed(a, parent())
    c1 = feed(a, child("c1", "s1", t_received=3), first_text=PROMPT)
    c2 = feed(a, child("c2", "s2", t_received=3), first_text="something unrelated to the arguments here")
    assert c1["subagent_edge"]["confidence"] == "high" and c1["subagent_edge"]["tool_call_id"] == "toolu_agent"
    assert c2["parent_call_id"] == "p1" and c2["subagent_edge"]["tool_call_id"] is None
    assert not c2.get("unattached")


def test_nested_subagent_attaches_to_the_subagents_call_not_the_outer_one():
    """S1 calls Agent (t1). Its child S2 calls Agent (t2) to spawn S3. While S3 starts, t1 and t2 are both open."""
    a = SubagentAttacher()
    feed(a, parent("p1", "S1", t_end=2, tools=[agent("t1", "Audit the whole repository for dead code")]))
    feed(a, child("c2", "S2", t_received=3), first_text="Audit the whole repository for dead code")
    s2_call = parent("c2b", "S2", t_received=4, t_end=5, tools=[agent("t2", "Check module utils.py for unused functions")])
    s2_call["step_index"] = 1
    feed(a, s2_call)
    c3 = feed(a, child("c3", "S3", t_received=6), first_text="Check module utils.py for unused functions")
    assert c3["subagent_edge"]["parent_call_id"] == "c2b" and c3["subagent_edge"]["confidence"] == "high"


def test_nested_without_echo_still_picks_the_subagents_own_call():
    a = SubagentAttacher()
    feed(a, parent("p1", "S1", t_end=2, tools=[agent("t1", "Audit the whole repository for dead code")]))
    feed(a, child("c2", "S2", t_received=3), first_text="Audit the whole repository for dead code")
    s2_call = parent("c2b", "S2", t_received=4, t_end=5, tools=[{"id": "t2", "name": "Agent", "args": "{}"}])
    s2_call["step_index"] = 1
    feed(a, s2_call)
    c3 = feed(a, child("c3", "S3", t_received=6), first_text="prompt that the arguments do not contain")
    assert c3["subagent_edge"]["parent_call_id"] == "c2b"


def test_elimination_never_applies_across_clients():
    a = SubagentAttacher()
    feed(a, parent("p1", key="alice", tools=[agent("t1", "Summarise the billing module in detail please")]))
    feed(a, child("c1", "s1", key="alice", t_received=3), first_text="Summarise the billing module in detail please")
    c = feed(a, child("c2", "s2", key="bob", t_received=3), first_text="unrelated work by another user entirely")
    assert c.get("parent_call_id") is None and not c.get("unattached")


# ---------------------------------------------------------------- real-world shape: harness boilerplate blocks
def test_harness_wrapper_blocks_are_skipped_when_finding_the_childs_prompt():
    """Found on real Claude Code: the child's first message is [system-reminder, instruction, system-reminder]."""
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>\nAttribution for git commits...\n</system-reminder>"},
        {"type": "text", "text": "Count the lines in api.py and report the number."},
        {"type": "text", "text": "<system-reminder>\nYour final report is delivered through X\n</system-reminder>"}]}]}
    assert first_human_text("anthropic", body) == "Count the lines in api.py and report the number."


def test_only_wrapper_blocks_falls_back_to_them_rather_than_nothing():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>x</system-reminder>"}]}]}
    assert first_human_text("anthropic", body) == "<system-reminder>x</system-reminder>"


def test_a_real_prompt_that_merely_contains_a_tag_is_not_dropped():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Fix this: <div>hello</div> and then explain why it broke."}]}]}
    assert "Fix this" in first_human_text("anthropic", body)


def test_three_parallel_children_with_boilerplate_blocks_each_find_their_own_call():
    a = SubagentAttacher()
    names = ["api.py", "sessions.py", "subagents.py"]
    feed(a, parent(tools=[agent(f"t{i}", f"You are a read-only helper for the project. Report how many lines {n} has.")
                          for i, n in enumerate(names)]))
    wrap = {"type": "text", "text": "<system-reminder>\nsame boilerplate for everyone\n</system-reminder>"}
    edges = []
    for i, n in enumerate(names):
        body = {"messages": [{"role": "user", "content": [wrap, {"type": "text", "text":
                f"You are a read-only helper for the project. Report how many lines {n} has."}, wrap]}]}
        c = feed(a, child(f"c{i}", f"s{i}", t_received=3), first_text=first_human_text("anthropic", body))
        edges.append(c["subagent_edge"])
    assert [e["confidence"] for e in edges] == ["high"] * 3
    assert sorted(e["tool_call_id"] for e in edges) == ["t0", "t1", "t2"]


# ---------------------------------------------------------------- one parent call, many helpers
def test_every_helper_started_by_one_call_attaches_to_that_call():
    """The rule: all helpers issued by one model call hang off that call, whatever else is going on."""
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", "Summarise the billing module in detail please"),
                          agent("t2", "List every TODO in the refunds module please"),
                          {"id": "t3", "name": "Read", "args": json.dumps({"file_path": "x.py"})}]))
    kids = [feed(a, child("c1", "s1", t_received=3), first_text="Summarise the billing module in detail please"),
            feed(a, child("c2", "s2", t_received=3), first_text="List every TODO in the refunds module please"),
            feed(a, child("c3", "s3", t_received=3), first_text="a third helper whose prompt is not in any argument")]
    assert [k["parent_call_id"] for k in kids] == ["p1", "p1", "p1"]
    assert [k["subagent_edge"]["tool_call_id"] for k in kids[:2]] == ["t1", "t2"]
    assert not any(k.get("unattached") for k in kids) and a.unattached == 0


def test_last_sibling_narrows_to_the_remaining_tool_by_elimination():
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", "Summarise the billing module in detail please"),
                          {"id": "t2", "name": "Agent", "args": "{}"}]))
    feed(a, child("c1", "s1", t_received=3), first_text="Summarise the billing module in detail please")
    c2 = feed(a, child("c2", "s2", t_received=3), first_text="a prompt that is nowhere in the arguments")
    assert c2["parent_call_id"] == "p1" and c2["subagent_edge"]["tool_call_id"] == "t2"
    assert c2["subagent_edge"]["signals"] == ["overlap", "elimination"]


def test_more_helpers_than_tool_calls_still_attach_to_the_call():
    a = SubagentAttacher()
    feed(a, parent(tools=[agent("t1", "Summarise the billing module in detail please")]))
    feed(a, child("c1", "s1", t_received=3), first_text="Summarise the billing module in detail please")
    extra = feed(a, child("c2", "s2", t_received=3), first_text="an extra helper nobody can account for")
    assert extra["parent_call_id"] == "p1" and extra["subagent_edge"]["tool_call_id"] is None


def test_different_steps_open_at_once_with_no_echo_is_still_not_guessed():
    a = SubagentAttacher()
    feed(a, parent("p_old", "s1", t_end=2, tools=[{"id": "t_old", "name": "Agent", "args": "{}"}]))
    feed(a, parent("p_new", "s2", t_end=4, tools=[{"id": "t_new", "name": "Agent", "args": "{}"}]))
    c = feed(a, child(t_received=6), first_text="nothing in common with either call")
    assert c.get("parent_call_id") is None and c["unattached"] is True and c["overlap_candidates"] == 2
