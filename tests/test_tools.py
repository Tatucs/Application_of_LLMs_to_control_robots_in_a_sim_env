"""Tests for the Panda MCP server.

Unit tests for the helpers run anywhere. The end-to-end tests drive the real
arm through an MCP stdio client and are skipped unless Webots is running.
"""
import asyncio
import base64
import json
import math
import os
import time
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from llmbot import panda_mcp_server as server

SERVER = Path(__file__).resolve().parents[1] / "src" / "llmbot" / "panda_mcp_server.py"
TOOLS = {"get_gripper_state", "get_objects", "get_target_position", "pick", "place", "go_home",
         "get_camera_image"}
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


# ----------------------------------------- request layer, fake status.json, no Webots
@pytest.fixture
def fake_link(tmp_path, monkeypatch):
    """Point the server at a temp dir, so a test can write status.json itself."""
    monkeypatch.setattr(server, "STATUS_FILE", tmp_path / "status.json")
    monkeypatch.setattr(server, "COMMANDS_FILE", tmp_path / "commands.json")
    return tmp_path


def test_no_status_file_is_timeout(fake_link):
    r = server.run_command("get_gripper_state")
    assert r["reason"] == "timeout" and "not running" in r["detail"]


def test_stale_heartbeat_is_timeout(fake_link):
    # plan row 12: a paused sim stops the heartbeat
    old = time.time() - 2 * server.HEARTBEAT_MAX_DELAY
    server.write_json(server.STATUS_FILE, {"heartbeat": {"wall_time": old}, "busy_with": None})
    start = time.monotonic()
    r = server.run_command("get_gripper_state")
    assert r["reason"] == "timeout" and "not running" in r["detail"]
    assert time.monotonic() - start < 3.0
    assert not server.COMMANDS_FILE.exists(), "no command may be sent to a dead sim"


def test_busy_controller_is_busy(fake_link):
    server.write_json(server.STATUS_FILE,
                      {"heartbeat": {"wall_time": time.time()}, "busy_with": "other_id"})
    r = server.run_command("get_gripper_state")
    assert r["reason"] == "busy"
    assert not server.COMMANDS_FILE.exists(), "no command may be sent while busy"


# ----------------------------------------- camera layer, fake recognition.json, no Webots
RED_CUBE = {"id": 370, "colours": [1.0, 0.0, 0.0],
            "position_world": [0.4588, -0.586, 0.0198], "yaw_world": -0.0}


@pytest.fixture
def fake_camera(tmp_path, monkeypatch):
    """Point the server at a temp dir, so a test can write recognition.json and image.jpg itself."""
    monkeypatch.setattr(server, "RECOGNITION_FILE", tmp_path / "recognition.json")
    monkeypatch.setattr(server, "IMAGE_FILE", tmp_path / "image.jpg")
    return tmp_path


def write_frame(objects: list, age: float = 0.0) -> None:
    """Write a camera frame taken age seconds ago (wall time)."""
    server.write_json(server.RECOGNITION_FILE,
                      {"time": 1.0, "wall_time": time.time() - age, "objects": objects})


@pytest.mark.parametrize("rgb, word", [([1, 0, 0], "red"), ([0.9, 0.1, 0.1], "red"),
                                       ([1, 1, 0.2], "yellow"), ([], "unknown")])
def test_colour_word_nearest(rgb, word):
    assert server.colour_word(rgb) == word


def test_no_recognition_file_is_timeout(fake_camera):
    start = time.monotonic()
    r = server.get_objects()
    assert r["reason"] == "timeout" and "not running" in r["detail"]
    assert time.monotonic() - start < 1.0


def test_stale_frame_is_timeout(fake_camera):
    # plan test 9: a paused sim or a dead camera stops new frames
    write_frame([RED_CUBE], age=2 * server.FRAME_MAX_AGE)
    for r in (server.get_objects(), server.get_target_position("obj_370")):
        assert r["reason"] == "timeout" and "paused" in r["detail"]


def test_get_objects_from_frame(fake_camera):
    write_frame([RED_CUBE, {**RED_CUBE, "id": 371, "colours": []}])
    r = server.get_objects()
    check_clean(r)
    assert r == {"ok": True, "reason": "ok",
                 "objects": [{"id": "obj_370", "colour": "red"},
                             {"id": "obj_371", "colour": "unknown"}]}


def test_get_objects_empty_scene(fake_camera):
    write_frame([])
    assert server.get_objects() == {"ok": True, "reason": "ok", "objects": []}


def test_get_target_position_from_frame(fake_camera):
    write_frame([RED_CUBE])
    r = server.get_target_position(" obj_370 ")
    check_clean(r)                                   # yaw -0.0 must come out as 0.0
    assert r == {"ok": True, "reason": "ok", "object_id": "obj_370",
                 "x": 0.459, "y": -0.586, "z": 0.02, "yaw": 0.0}


def test_get_target_position_not_visible(fake_camera):
    # an id from an earlier get_objects whose object is now hidden
    write_frame([RED_CUBE])
    r = server.get_target_position("obj_999")
    assert r["reason"] == "not_found" and "obj_999" in r["detail"]


JPEG = b"\xff\xd8\xff\xe0fake jpeg\xff\xd9"


def test_camera_image_returns_dict_and_image(fake_camera):
    server.IMAGE_FILE.write_bytes(JPEG)
    status, image = server.get_camera_image()
    assert status == {"ok": True, "reason": "ok"}
    assert image.data == JPEG


def test_no_image_file_is_timeout(fake_camera):
    r = server.get_camera_image()
    assert len(r) == 1 and r[0]["reason"] == "timeout" and "not running" in r[0]["detail"]


def test_stale_image_is_timeout(fake_camera):
    server.IMAGE_FILE.write_bytes(JPEG)
    old = time.time() - 2 * server.FRAME_MAX_AGE
    os.utime(server.IMAGE_FILE, (old, old))
    r = server.get_camera_image()
    assert len(r) == 1 and r[0]["reason"] == "timeout" and "paused" in r[0]["detail"]


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
    tools = await client.list_tools()
    assert {t.name for t in tools} == TOOLS
    assert all(t.description for t in tools), "every tool needs a docstring"


@needs_sim
async def test_every_object_has_a_position(client):
    objs = await tool(client, "get_objects")
    for o in objs["objects"]:
        p = await tool(client, "get_target_position", {"object_id": o["id"]})
        assert p["ok"] and p["object_id"] == o["id"], p


@needs_sim
async def test_camera_matches_supervisor(client):
    """Plan test 3: every L1 (camera) position within 5 mm and yaw within 0.01 rad
    of the L0 (Supervisor) ground truth, which the controller still answers."""
    truth = server.run_command("get_objects", timeout=5.0)
    assert truth["ok"], truth
    l0 = [server.run_command("get_target_position", {"object_id": o["id"]}, timeout=5.0)
          for o in truth["objects"]]
    objs = await tool(client, "get_objects")
    assert len(objs["objects"]) == len(l0), "camera and Supervisor see a different number of objects"
    for o in objs["objects"]:
        p = await tool(client, "get_target_position", {"object_id": o["id"]})
        nearest = min(l0, key=lambda q: math.dist((p["x"], p["y"], p["z"]), (q["x"], q["y"], q["z"])))
        err = math.dist((p["x"], p["y"], p["z"]), (nearest["x"], nearest["y"], nearest["z"]))
        assert err < 0.005, f"{o['id']} is {err * 1000:.1f} mm from {nearest['object_id']}"
        dyaw = (p["yaw"] - nearest["yaw"] + math.pi / 4) % (math.pi / 2) - math.pi / 4
        assert abs(dyaw) < 0.01, f"{o['id']} yaw off by {dyaw:.3f} rad (mod 90 deg, cube symmetry)"


@needs_sim
async def test_camera_image_over_mcp(client):
    """The client gets two blocks: the status as JSON text, then a real JPEG."""
    r = await client.call_tool("get_camera_image", {})
    text, image = r.content
    assert json.loads(text.text) == {"ok": True, "reason": "ok"}
    assert image.mime_type == "image/jpeg"
    assert base64.b64decode(image.data)[:2] == b"\xff\xd8"   # JPEG start-of-image marker


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
    assert (await tool(client, "get_gripper_state"))["holding"] is False


@needs_sim
async def test_pick_empty_spot_grasp_failed(client):
    xyz = await cube_position(client)
    empty = {**xyz, "x": xyz["x"] - 0.10}        # 10 cm from the cube, nothing there
    try:
        r = await tool(client, "pick", empty)
        assert r["reason"] == "grasp_failed" and r["stage"] == "close", r
        assert r["holding"] is False
    finally:
        await tool(client, "go_home")


@needs_sim
async def test_pick_out_of_reach_unreachable(client):
    xyz = await cube_position(client)
    far = {**xyz, "x": xyz["x"] + 1.5}           # Panda reach is ~0.85 m
    try:
        r = await tool(client, "pick", far)
        assert r["reason"] == "unreachable" and r["stage"] == "approach", r
    finally:
        await tool(client, "go_home")


@needs_sim
async def test_second_client_gets_busy(client):
    """Plan row 13: while one client's pick runs, another client's call is refused."""
    xyz = await cube_position(client)
    empty = {**xyz, "x": xyz["x"] - 0.10}        # a long pick that does not move the cube
    async with Client(SERVER) as other:
        running = asyncio.create_task(tool(client, "pick", empty))
        try:
            await asyncio.sleep(0.5)             # let the controller pick up the command
            r = await tool(other, "get_gripper_state")
            assert r["reason"] == "busy", r
        finally:
            await running
            await tool(client, "go_home")
