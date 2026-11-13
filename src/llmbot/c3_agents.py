'''
C3a closed-loop agent: C2 (planner + executor) plus a scene check and replanning.

After each execution, check_scene compares the scene with the plan through the
MCP perception tools; fidelity compares the executor's tool calls with the plan.
'''

import asyncio
import sys
from pathlib import Path
import time

from fastmcp.client.transports import StdioTransport
from pydantic import BaseModel
from pydantic_ai import RunUsage, capture_run_messages
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

# Constants
TOL = 0.02                      # m; max x/y distance of a cube from its place target
MAX_ATTEMPTS = 3                # max attempts to satisfy a task 

# MCP toolset, shared by the planner, the executor and the check
transport = StdioTransport(
    command=sys.executable,
    args=[str(Path(__file__).resolve().parent / "panda_mcp_server.py")]
)
toolset = MCPToolset(transport)


# ----------------------------------------- Temporary functions
# Temporary copy from Tomi's planner_executor.py; delete after the merge.

class Step(BaseModel):
    """One tool call of the plan."""
    step: int                          # 1, 2, 3, ...
    tool: str                          # "pick", "place" or "go_home"
    object_id: str | None = None       # pick AND place: which cube (from get_objects)
    target_area: str | None = None     # place: which area
    x: float | None = None             # pick/place pose; None for go_home
    y: float | None = None
    z: float | None = None
    yaw: float = 0.0
    note: str = ""                     # free text for the executor
    status: str | None = None          # executor writes "done" / "failed"


class Plan(BaseModel):
    """Output of the planner, input of the executor."""
    reasoning: str                     # keep this, it's valuable for the case studies
    steps: list[Step]

async def make_plan(task: str, usage: RunUsage, previous: Plan | None = None) -> Plan:
    pass

async def execute_plan(plan: Plan, usage: RunUsage) -> Plan:
    pass


# ----------------------------------------- scene check

class CheckResult(BaseModel):
    ok: bool
    mismatches: list[str]              # e.g. "obj_412: 0.031 m from place target"
    error: str | None = None           # scene not visible (e.g. sim paused); stops the loop


async def check_scene(plan: Plan, attempt: int) -> CheckResult:
    """Check if every placed cube is at its place coordinates."""
    places = [s for s in plan.steps if s.tool == "place"]

    # No place steps: nothing was done on attempt 1;
    # after attempt 1 an empty plan means the task is already satisfied.
    if not places:
        if attempt == 1:
            return CheckResult(ok=False, mismatches=["no_place_steps"])
        return CheckResult(ok=True, mismatches=[])

    mismatches = []
    for s in places:
        if s.object_id is None or s.x is None or s.y is None:
            mismatches.append(f"step {s.step}: place step without object_id or x/y")
            continue

        r = await toolset.direct_call_tool("get_target_position", {"object_id": s.object_id})

        if r["reason"] == "not_found":
            mismatches.append(f"{s.object_id}: not found")
            continue
        if not r["ok"]:
            # timeout or error: the scene could not be seen, so no verdict
            return CheckResult(ok=False, mismatches=mismatches,
                               error=f"{s.object_id}: {r['reason']} ({r.get('detail', '')})")

        dx = r["x"] - s.x
        dy = r["y"] - s.y
        if abs(dx) > TOL or abs(dy) > TOL:
            dist = (dx**2 + dy**2) ** 0.5
            mismatches.append(f"{s.object_id}: {dist:.3f} m from place target")

    return CheckResult(ok=not mismatches, mismatches=mismatches)


# ----------------------------------------- executor fidelity

class FidelityResult(BaseModel):
    ok: bool
    first_mismatch: int | None = None  # 1-based position where execution left the plan


def fidelity(plan: Plan, messages: list[ModelMessage]) -> FidelityResult:
    """Did the executor call exactly the planned tools, in order?"""
    planned = [(s.tool, s.x, s.y, s.z, s.yaw) for s in plan.steps]

    executed = []
    for m in messages:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, ToolCallPart) and p.tool_name != "final_result":
                    a = p.args_as_dict()   # args may arrive as dict or JSON string
                    executed.append((p.tool_name, a.get("x"), a.get("y"), a.get("z"),
                                     a.get("yaw", 0.0)))

    # The final go_home comes from the executor prompt, not from the plan
    if executed and executed[-1][0] == "go_home" and len(executed) > len(planned):
        executed.pop()

    for i, (p, e) in enumerate(zip(planned, executed, strict=False), start=1):
        if p != e:
            return FidelityResult(ok=False, first_mismatch=i)
    if len(planned) != len(executed):
        # one list is a prefix of the other: first missing or extra call
        return FidelityResult(ok=False, first_mismatch=min(len(planned), len(executed)) + 1)
    return FidelityResult(ok=True)

# ----------------------------------------- records

class AttemptRecord(BaseModel):
    attempt: int                           # 1-based
    plan: Plan | None = None               # None if make_plan raised
    plan_error: str | None = None
    report: Plan | None = None             # marked plan from the executor; None if it raised
    exec_error: str | None = None
    check: CheckResult | None = None       # None if make_plan raised (no check, §3.2)
    fidelity: FidelityResult | None = None
    wall_time: float = 0.0                 # s, perf_counter


class TaskRecord(BaseModel):
    task: str
    warm: bool
    planner_model: str
    executor_model: str
    max_attempts: int
    attempts: list[AttemptRecord] = []
    success: bool = False
    planner_usage: RunUsage = RunUsage()
    executor_usage: RunUsage = RunUsage()

# ----------------------------------------- loop

async def run_task(task: str, warm: bool) -> TaskRecord:
    """C3 loop: plan, execute, check; replan until ok, check error, or MAX_ATTEMPTS."""
    rec = TaskRecord(task=task, warm=warm, planner_model="stub", executor_model="stub",
                     max_attempts=MAX_ATTEMPTS)
    previous = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        a = AttemptRecord(attempt=attempt)
        rec.attempts.append(a)
        t0 = time.perf_counter()

        # plan; a raise = failed attempt, no check
        try:
            a.plan = await make_plan(task, rec.planner_usage, previous)
        except Exception as e:
            a.plan_error = f"{type(e).__name__}: {e}"
            a.wall_time = time.perf_counter() - t0
            continue

        # execute; a raise = failed attempt, but still check
        with capture_run_messages() as msgs:
            try:
                a.report = await execute_plan(a.plan, rec.executor_usage)
            except Exception as e:
                a.exec_error = f"{type(e).__name__}: {e}"

        a.fidelity = fidelity(a.plan, msgs)
        a.check = await check_scene(a.plan, attempt)
        a.wall_time = time.perf_counter() - t0

        if a.check.error:        # scene not visible: stop, run discarded
            break
        if a.check.ok:
            rec.success = True
            break

        # C3b: send back the plan with only the executor's marks copied onto it
        if warm:
            previous = a.plan.model_copy(deep=True)
            marks = {s.step: s.status for s in a.report.steps} if a.report else {}
            for s in previous.steps:
                s.status = marks.get(s.step)

    return rec

# ----------------------------------------- main

async def main():
    """Hand test of check_scene on the first graspable cube (sim must be running)."""
    async with toolset:
        objs = await toolset.direct_call_tool("get_objects", {})
        print(objs)
        cube = next(o["id"] for o in objs["objects"] if o["kind"] == "graspable")
        pos = await toolset.direct_call_tool("get_target_position", {"object_id": cube})
        print(pos)

        plan = Plan(reasoning="hand test", steps=[
            Step(step=1, tool="place", object_id=cube, x=pos["x"], y=pos["y"], z=pos["z"]),
        ])
        print(await check_scene(plan, attempt=1))   # expect ok
        plan.steps[0].x += 0.05
        print(await check_scene(plan, attempt=1))   # expect 0.050 m mismatch


if __name__ == "__main__":
    asyncio.run(main())
