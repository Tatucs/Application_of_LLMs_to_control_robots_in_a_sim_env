"""Tests for the Panda MCP server.

Unit tests for the helpers run anywhere. The end-to-end tests drive the real
arm through an MCP stdio client and are skipped unless Webots is running.
"""
import math
import time
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from llmbot import panda_mcp_server as server

SERVER = Path(__file__).resolve().parents[1] / "src" / "llmbot" / "panda_mcp_server.py"
TOOLS = {"get_gripper_state", "get_objects", "get_target_position", "pick", "place", "go_home"}
FORBIDDEN = {"id", "left_mm", "right_mm", "force_n", "target", "reached", "pos_err_mm"}


def sim_running() -> bool:
    """True if the controller's heartbeat is fresh, same check as call()."""
    beat = (server.read_json(server.STATUS_FILE) or {}).get("heartbeat") or {}
    return time.time() - beat.get("wall_time", 0) < server.HEARTBEAT_MAX_DELAY


needs_sim = pytest.mark.skipif(not sim_running(), reason="Webots not running or paused")


def check_clean(data, top: bool = True):
    """Fail if a debug key (top level only) or -0.0 (anywhere) got through trim()."""
    if isinstance(data, dict):
        if top:
            assert not FORBIDDEN & data.keys(), f"untrimmed keys {FORBIDDEN & data.keys()}"
        for v in data.values():
            check_clean(v, top=False)
    elif isinstance(data, list):
        for v in data:
            check_clean(v, top=False)
    elif isinstance(data, float):
        assert math.copysign(1, data) > 0 or data != 0.0, "-0.0 found"


# ----------------------------------------- unit tests, no Webots
def test_clean_rounds_and_fixes_negative_zero():
    assert server.clean(0.123456) == 0.123
    assert math.copysign(1, server.clean(-0.0)) > 0
    assert server.clean({"a": [-0.0, 1.23456], "b": "x"}) == {"a": [0.0, 1.235], "b": "x"}


def test_trim_keeps_only_whitelisted_keys():
    raw = {"ok": True, "reason": "ok", "id": "abc", "left_mm": 1.0, "holding": True}
    assert server.trim(raw) == {"ok": True, "reason": "ok", "holding": True}


def test_trim_keeps_nested_object_ids():
    raw = {"ok": True, "reason": "ok", "objects": [{"id": "cube", "colour": "white"}]}
    assert server.trim(raw)["objects"] == [{"id": "cube", "colour": "white"}]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_not_finite_rejects(bad):
    err = server.not_finite(x=0.5, y=bad)
    assert err["reason"] == "error" and "y" in err["detail"]


def test_not_finite_accepts_normal_numbers():
    assert server.not_finite(x=0.5, y=0.0, z=-0.1) is None


# ----------------------------------------- end-to-end, needs Webots
@pytest.fixture
async def client():
    """A fresh MCP client; it starts its own server process over stdio."""
    async with Client(SERVER) as c:
        yield c


async def tool(c: Client, name: str, args: dict | None = None) -> dict:
    """Call a tool and check that its result is trimmed."""
    r = (await c.call_tool(name, args or {})).data
    check_clean(r)
    return r


async def cube_position(c: Client) -> dict:
    """x, y, z, yaw of the first object in the scene."""
    objs = await tool(c, "get_objects")
    assert objs["ok"] and objs["objects"], "no objects in the scene"
    p = await tool(c, "get_target_position", {"object_id": objs["objects"][0]["id"]})
    assert p["ok"]
    return {k: p[k] for k in ("x", "y", "z", "yaw")}


@needs_sim
async def test_lists_all_tools(client):
    assert {t.name for t in await client.list_tools()} == TOOLS


@needs_sim
async def test_gripper_starts_empty(client):
    r = await tool(client, "get_gripper_state")
    assert r["ok"], r.get("detail")
    assert r["holding"] is False, "start with an empty gripper (call go_home)"


@needs_sim
async def test_get_target_position_empty_id(client):
    r = await tool(client, "get_target_position", {"object_id": "   "})
    assert r["reason"] == "error"


@needs_sim
async def test_get_target_position_unknown_id(client):
    r = await tool(client, "get_target_position", {"object_id": "no_such_cube"})
    assert r["reason"] == "not_found"


@needs_sim
async def test_pick_nan_rejected_by_schema(client):
    # NaN is sent as JSON null, so pydantic rejects it before the tool runs
    xyz = await cube_position(client)
    with pytest.raises(ToolError):
        await client.call_tool("pick", {**xyz, "z": math.nan})


@needs_sim
async def test_place_with_empty_gripper(client):
    r = await tool(client, "place", await cube_position(client))
    assert r["reason"] == "not_holding" and r["stage"] == "precheck"


@needs_sim
async def test_pick_place_cycle(client):
    """Pick the cube, try a second pick, put it back, go home."""
    xyz = await cube_position(client)
    try:
        r = await tool(client, "pick", xyz)
        assert r["ok"] and r["holding"], f"pick failed: {r}"

        r = await tool(client, "pick", xyz)
        assert r["reason"] == "already_holding"

        r = await tool(client, "place", xyz)
        assert r["ok"], f"place failed: {r}"
    finally:
        r = await tool(client, "go_home")        # always leave the arm at home
    assert r["ok"] and r["dropped"] is False
