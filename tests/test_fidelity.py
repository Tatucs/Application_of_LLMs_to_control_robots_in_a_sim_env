"""fidelity on hand-written executor messages. Offline: no LLM, no sim.

    pytest tests/test_fidelity.py -v
"""

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart

from llmbot.c3a_closed_loop_agent import Plan, Step, fidelity

PLAN = Plan(reasoning="test", steps=[
    Step(step=1, tool="pick", object_id="obj_1", x=0.5, y=0.1, z=0.05),
    Step(step=2, tool="place", object_id="obj_1", x=0.3, y=-0.2, z=0.1),
])

PICK = ("pick", {"x": 0.5, "y": 0.1, "z": 0.05})
PLACE = ("place", {"x": 0.3, "y": -0.2, "z": 0.1})
HOME = ("go_home", {})


def calls(*cs):
    """One ModelResponse per tool call, as the executor produces them."""
    return [ModelResponse(parts=[ToolCallPart(tool_name=n, args=a)]) for n, a in cs]


@pytest.mark.parametrize("msgs, ok, first_mismatch", [
    (calls(PICK, PLACE), True, None),                                           # exact
    (calls(PICK, PLACE, HOME), True, None),                                     # final go_home dropped
    (calls(("pick", '{"x":0.5,"y":0.1,"z":0.05,"yaw":0.0}'), PLACE), True, None),  # args as JSON string
    (calls(PICK, ("place", {"x": 0.31, "y": -0.2, "z": 0.1})), False, 2),       # wrong args
    (calls(("place", PICK[1]), PLACE), False, 1),                               # wrong tool
    (calls(PICK), False, 2),                                                    # missing step
    (calls(PICK, HOME), False, 2),                                              # stopped, then go_home
    (calls(PICK, PICK, PLACE), False, 2),                                       # retry
    (calls(PICK, PLACE, PICK), False, 3),                                       # extra call
    ([], False, 1),                                                             # nothing executed
], ids=["exact", "final_go_home", "json_args", "wrong_args", "wrong_tool",
        "missing", "stopped", "retry", "extra_call", "nothing"])
def test_fidelity(msgs, ok, first_mismatch):
    r = fidelity(PLAN, msgs)
    assert (r.ok, r.first_mismatch) == (ok, first_mismatch)
