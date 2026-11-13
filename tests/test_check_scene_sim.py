"""check_scene against the live Webots sim (Panda world).

Not offline tests: they need Webots running, and the last two move the arm.

    pytest tests/test_check_scene_sim.py -v -s          # sim running: tests 1, 3, 4, 5
    pytest tests/test_check_scene_sim.py -v -s -k paused  # sim paused: test 2 only

Every test skips if the sim is in the wrong state for it, so the file never
fails just because Webots is not running.
"""

import pytest
import pytest_asyncio

from llmbot.c3a_closed_loop_agent import TOL, Plan, Step, check_scene, toolset

# `toolset` is a module-level object, so it stays bound to the event loop it
# first connected on. pytest-asyncio's default (a new loop per test) closes that
# loop after test 1 -> "Event loop is closed" from test 2 on. One loop per module.
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def call(tool: str, args: dict | None = None) -> dict:
    return await toolset.direct_call_tool(tool, args or {})


@pytest_asyncio.fixture(loop_scope="module")
async def sim():
    """Open the MCP server; skip unless the camera answers (sim running)."""
    async with toolset:
        objs = await call("get_objects")
        if not objs["ok"]:
            pytest.skip(f"sim not running: {objs['reason']}")
        yield objs


def cube_ids(objs: dict) -> list[str]:
    return [o["id"] for o in objs["objects"] if o["kind"] == "graspable"]


def area_id(objs: dict) -> str:
    return next(o["id"] for o in objs["objects"] if o["kind"] == "area")


# --- 1. unknown id -> mismatch "not found"
async def test_unknown_id_is_not_found(sim):
    plan = Plan(reasoning="", steps=[
        Step(step=1, tool="place", object_id="obj_999", x=0.5, y=0.0, z=0.02),
    ])
    r = await check_scene(plan, attempt=1)
    print(r)
    assert not r.ok
    assert r.error is None
    assert r.mismatches == ["obj_999: not found"]


# --- 2. paused sim -> error, no verdict (run with the sim PAUSED, -k paused)
async def test_paused_sim_sets_error():
    async with toolset:
        if (await call("get_objects"))["ok"]:
            pytest.skip("sim is running; pause it and run with -k paused")
        plan = Plan(reasoning="", steps=[
            Step(step=1, tool="place", object_id="obj_1", x=0.5, y=0.0, z=0.02),
        ])
        r = await check_scene(plan, attempt=1)
        print(r)
        assert not r.ok
        assert r.error is not None
        assert "timeout" in r.error


# --- 3. empty plan: failure on attempt 1, success on a replan
async def test_empty_plan(sim):
    empty = Plan(reasoning="", steps=[])
    r1 = await check_scene(empty, attempt=1)
    r2 = await check_scene(empty, attempt=2)
    print(r1, r2, sep="\n")
    assert not r1.ok and r1.mismatches == ["no_place_steps"]
    assert r2.ok and r2.mismatches == []


# --- 4. home pose does not hide any cube (MOVES THE ARM)
async def test_home_pose_occludes_nothing(sim):
    before = cube_ids(sim)
    home = await call("go_home")
    assert home["ok"], home
    after = cube_ids(await call("get_objects"))
    print("before:", before, "\nafter: ", after)
    assert sorted(after) == sorted(before)


# --- 5. measure place error for TOL (MOVES CUBES INTO THE AREA)
async def test_measure_place_error(sim):
    area = await call("get_area_position", {"object_id": area_id(sim)})
    assert area["ok"], area
    # three well-separated points inside the safe region of the area
    targets = [
        (area["x"], area["y"]),
        (area["x_min"], area["y_min"]),
        (area["x_max"], area["y_max"]),
    ]

    steps = []
    for i, (cube, (tx, ty)) in enumerate(zip(cube_ids(sim), targets, strict=False)):
        pos = await call("get_target_position", {"object_id": cube})
        assert pos["ok"], pos
        picked = await call("pick", {k: pos[k] for k in ("x", "y", "z", "yaw")})
        assert picked["ok"], picked
        placed = await call("place", {"x": tx, "y": ty, "z": area["z"]})
        assert placed["ok"], placed
        steps.append(Step(step=i + 1, tool="place", object_id=cube, x=tx, y=ty, z=area["z"]))

    home = await call("go_home")  # arm out of the camera image before measuring
    assert home["ok"], home

    errors = []
    for s in steps:
        pos = await call("get_target_position", {"object_id": s.object_id})
        assert pos["ok"], pos
        dx, dy = pos["x"] - s.x, pos["y"] - s.y
        errors.append(max(abs(dx), abs(dy)))
        print(f"{s.object_id}: dx={dx:+.4f} dy={dy:+.4f} m")
    print(f"max per-axis error {max(errors):.4f} m, TOL {TOL} m")

    r = await check_scene(Plan(reasoning="", steps=steps), attempt=1)
    print(r)
    assert r.ok, "real places fail the check: TOL too small or place inaccurate"
