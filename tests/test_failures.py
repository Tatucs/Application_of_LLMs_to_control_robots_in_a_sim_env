"""failures on hand-written executor messages. Offline: no LLM, no sim.

    pytest tests/test_failures.py -v
"""

from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from llmbot.c3_agents import Failure, Plan, Step, failures

PLAN = Plan(reasoning="test", steps=[
    Step(step=1, tool="pick", object_id="obj_1", x=0.5, y=0.1, z=0.05),
    Step(step=2, tool="place", object_id="obj_1", x=0.3, y=-0.2, z=0.1),
])

OK = {"ok": True}
UNREACHABLE = {"ok": False, "reason": "unreachable", "detail": "outside workspace"}
TIMEOUT = {"ok": False, "reason": "timeout", "detail": "simulation not running or paused"}


def run(*cs):
    """Executor messages: per call a ModelResponse with the call, then a ModelRequest
    with its result. result=None means no result came back (executor raised)."""
    msgs = []
    for i, (tool, result) in enumerate(cs):
        cid = f"call_{i}"
        msgs.append(ModelResponse(parts=[ToolCallPart(tool_name=tool, args={}, tool_call_id=cid)]))
        if result is not None:
            msgs.append(ModelRequest(parts=[ToolReturnPart(tool_name=tool, content=result,
                                                           tool_call_id=cid)]))
    return msgs


def test_no_failures():
    assert failures(PLAN, run(("pick", OK), ("place", OK)), attempt=1) == []


def test_one_failure():
    assert failures(PLAN, run(("pick", OK), ("place", UNREACHABLE)), attempt=2) == [
        Failure(attempt=2, step=2, tool="place", object_id="obj_1",
                reason="unreachable", detail="outside workspace"),
    ]


def test_no_result_skipped():
    # pick failed, then the executor raised during place: place has no result
    r = failures(PLAN, run(("pick", UNREACHABLE), ("place", None)), attempt=1)
    assert [(f.step, f.reason) for f in r] == [(1, "unreachable")]


def test_off_plan_call_skipped():
    # go_home where the plan says pick: that pair is skipped
    r = failures(PLAN, run(("go_home", UNREACHABLE), ("place", TIMEOUT)), attempt=1)
    assert [(f.step, f.reason) for f in r] == [(2, "timeout")]


def test_timeout_kept():
    r = failures(PLAN, run(("pick", TIMEOUT)), attempt=1)
    assert [(f.step, f.reason, f.detail) for f in r] == [
        (1, "timeout", "simulation not running or paused"),
    ]


def test_nothing_executed():
    assert failures(PLAN, [], attempt=1) == []
