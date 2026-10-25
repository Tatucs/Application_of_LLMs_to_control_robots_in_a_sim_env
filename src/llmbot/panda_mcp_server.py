'''
MCP server for controlling the Panda robot in Webots.

This server works in conjunction with the Webots controller (panda_mcp_controller.py)
via the file system (commands.json / status.json) to exchange commands and status.
'''

import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path

from fastmcp import FastMCP

# Define paths for data exchange with the controller
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "panda"
CAMERA_DIR = Path(__file__).resolve().parents[2] / "data" / "camera"
COMMANDS_FILE = DATA_DIR / "commands.json"
STATUS_FILE = DATA_DIR / "status.json"
LOG_FILE = DATA_DIR / "server.log"
RECOGNITION_FILE = CAMERA_DIR / "recognition.json"

# Time constants
PICKUP_TIMEOUT = 2.0            # s wall time for the controller to accept a command
HEARTBEAT_MAX_DELAY = 3.0       # s wall time; older heartbeat = sim paused or stopped
POLL_PERIOD = 0.05              # s between status reads
FRAME_MAX_AGE = 3.0             # s wall time; older camera frame = camera dead or sim paused

# Keys the model sees; everything else is controller-side debug
KEEP = {"ok", "reason", "detail", "stage", "holding", "width_mm", "dropped",
        "objects", "object_id", "x", "y", "z", "yaw"}

# Colour words for recognitionColors
COLOURS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
           "yellow": (1, 1, 0), "white": (1, 1, 1)}

# Configure logging
logging.basicConfig(
    level=logging.INFO,             # change to DEBUG for more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding="utf-8", delay=True),
    ],
)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("Panda MCP Server")

# ----------------------------------------- FILE I/O
def write_json(path: Path, data: dict, retries: int = 20) -> None:
    """Write data to a JSON file atomically.

    Writes to a .tmp file first, then swaps it in with os.replace, so the
    controller never reads a half-written file. On Windows os.replace fails
    while the controller has the file open, so the swap is retried.

    Args:
        path: Target file, e.g. COMMANDS_FILE.
        data: JSON-serialisable dict to write.
        retries: Swap attempts before the last one, 10 ms apart.

    Returns:
        None.

    Raises:
        PermissionError: If the file is still locked after all retries.
        TypeError: If data is not JSON-serialisable.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows: controller has commands.json open right now
            time.sleep(0.01)
    os.replace(tmp, path)         # last try: real failure -> raise

def read_json(path: Path) -> dict | None:
    """Read and parse a JSON file without raising.

    Args:
        path: File to read, e.g. STATUS_FILE.

    Returns:
        The parsed content, or None if the file is missing, half-written
        or locked. None means "not available yet": the caller should
        poll again.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None

# ----------------------------------------- request / tool wrapper
def result_if_done(status: dict | None, cmd_id: str) -> dict | None:
    """Check if the controller has finished a command.
    
    Args:
        status: The current status of the controller.
        cmd_id: The ID of the command to check.

    Returns:
        The result of the command if it's done, or None if it's not.
    """
    result = (status or {}).get("last_result") or {}
    return result if result.get("id") == cmd_id else None

def run_command(action: str, args: dict | None = None, timeout: float = 60.0) -> dict:
    """Send one command to the controller and wait for its result.

    Args:
        action: Controller action name, e.g. "get_gripper_state".
        args: Keyword arguments for the action.
        timeout: Max wall-clock seconds to wait for the result once the
            controller has accepted the command.

    Returns:
        The controller's result dict (ok, reason, detail, ...), or an error
        dict with reason "busy", "timeout" or "error". Never raises.
    """
    # 1) check if the controller is running and not busy
    status = read_json(STATUS_FILE)
    if status is None:
        return {"ok": False, "reason": "timeout",
                "detail": "no status.json, controller not running"}
    if status.get("busy_with") is not None:
        return {"ok": False, "reason": "busy",
                "detail": f"controller busy with {status['busy_with']}"}
    beat = status.get("heartbeat") or {}
    if time.time() - beat.get("wall_time", 0) > HEARTBEAT_MAX_DELAY:
        return {"ok": False, "reason": "timeout", "detail": "simulation not running or paused"}

    # 2) write the command to the commands.json file
    cmd_id = uuid.uuid4().hex
    try:
        write_json(COMMANDS_FILE, {"id": cmd_id, "action": action, "args": args or {}})
    except OSError as e:
        return {"ok": False, "reason": "error", "detail": f"could not write command: {e}"}

    # 3) wait until the controller accepted the command
    deadline = time.time() + PICKUP_TIMEOUT
    while True:
        status = read_json(STATUS_FILE)
        if result := result_if_done(status, cmd_id):
            return result                                   # instant result (precheck)
        if status and status.get("busy_with") == cmd_id:
            break
        if time.time() > deadline:
            return {"ok": False, "reason": "timeout",
                    "detail": "controller did not pick up the command"}
        time.sleep(POLL_PERIOD)

    # 4) wait for the result
    deadline = time.time() + timeout
    while time.time() < deadline:
        if result := result_if_done(read_json(STATUS_FILE), cmd_id):
            return result
        time.sleep(POLL_PERIOD)
    return {"ok": False, "reason": "timeout",
            "detail": f"no result within {timeout} s, state unknown, call get_gripper_state"}

def call(action: str, args: dict | None = None, timeout: float = 60.0) -> dict:
    """Call a controller action and return its result. Also logs every call with its result.

    Args:
        action: Controller action name, e.g. "get_gripper_state".
        args: Keyword arguments for the action.
        timeout: Max wall-clock seconds to wait for the result once the
            controller has accepted the command.

    Returns:
        The controller's result dict (ok, reason, detail, ...), or an error
        dict with reason "busy", "timeout" or "error". Never raises.
    """
    start = time.monotonic()
    result = run_command(action, args=args, timeout=timeout)
    span = time.monotonic() - start
    logger.info(f"{action} {args or {}} -> {result.get('reason')} ({span:.3f} s)")
    logger.debug(f"{action} {args or {}} -> {result}")
    return result

# ----------------------------------------- camera
def read_frame() -> dict:
    """Read the latest camera frame from recognition.json without raising.
    
    Returns:
        ok, reason, objects (raw Recognition dicts), or an error dict.
    """
    retry = 0
    while retry < 5:
        frame = read_json(RECOGNITION_FILE)
        if frame is not None:
            frame_time = frame.get("wall_time", 0)
            if time.time() - frame_time > FRAME_MAX_AGE:
                return {"ok": False, "reason": "timeout",
                        "detail": "camera not running or simulation paused"}
            return {"ok": True, "reason": "ok", "objects": frame.get("objects", [])}
        retry += 1
        time.sleep(POLL_PERIOD)
    return {"ok": False, "reason": "timeout", "detail": "no recognition.json, camera not running"}

def colour_word(rgb: list[float]) -> str:
    """Return a colour word for a given RGB triple, or "unknown" if rgb is empty.

    Args:
        rgb: List of 3 floats in [0, 1].

    Returns:
        A colour word like "red", "green", "blue", "yellow", "white", or "unknown".
    """
    if len(rgb) < 3:
        return "unknown"
    return min(COLOURS, key=lambda c: math.dist(rgb[:3], COLOURS[c]))

# ----------------------------------------- MCP helpers
def clean(value):
    """Round floats to 3 decimals and turn -0.0 into 0.0.
    
    Args:
        value: A JSON value, nested lists and dicts allowed.

    Returns:
        The same structure with every float cleaned.
    """
    if isinstance(value, float):
        return round(value, 3) + 0.0
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value

def trim(result: dict) -> dict:
    """Keep only the keys in KEEP and clean the values.

    Args:
        result: Result dict from call().

    Returns:
        A smaller dict for the model, without id, debug values.
    """
    return {k: clean(v) for k, v in result.items() if k in KEEP}

def not_finite(**values: float) -> dict | None:
    """Check if any of the given values is NaN or infinite.

    Args:
        values: float value(s) to check.
    
    Returns:
        An error result if any value is not finite, or None if all are finite.
    """
    bad = [k for k, v in values.items() if not math.isfinite(v)]
    if bad:
        return {"ok": False, "reason": "error", "detail": f"non-finite values: {', '.join(bad)}"}
    return None

# ----------------------------------------- MCP tools
@mcp.tool()
def get_gripper_state() -> dict:
    """Report whether the gripper holds an object, and how wide it is open.
    
    Use before pick (gripper must be empty) or place (must be holding),
    and after a timeout, when the arm state is unknown.

    Returns:
        ok, reason, holding (bool), width_mm (opening between the
        fingers in mm).
    """
    return trim(call("get_gripper_state", timeout=5.0))

@mcp.tool()
def get_objects() -> dict:
    """List all graspable objects in the scene.

    Call this first: the ids are needed by get_target_position.
    If an object is not visible to the camera (e.g., it's under the arm), it will not be included in the list.

    Returns:
        ok, reason, objects (list of dicts with id, colour).
    """
    start = time.monotonic()
    frame = read_frame()
    span = time.monotonic() - start
    logger.info(f"get_objects -> {frame.get('reason')} ({span:.3f} s)")
    logger.debug(f"get_objects -> {frame}")
    if not frame["ok"]:
        return trim(frame)
    objects = [{"id": f"obj_{o['id']}", "colour": colour_word(o.get("colours", []))}
               for o in frame["objects"]]
    return trim({"ok": True, "reason": "ok", "objects": objects})

@mcp.tool()
def get_target_position(object_id: str) -> dict:
    """Get the centre and rotation of one object, ready to pass to pick.

    Args:
        object_id: An id from get_objects, e.g. "obj_412".

    Returns:
        ok, reason, object_id, x, y, z (metres, world frame), yaw (rad).
        reason "not_found" if the id does not exist or the camera does not see it.
    """
    object_id = object_id.strip()
    if not object_id:
        return {"ok": False, "reason": "error", "detail": "object_id is empty"}
    start = time.monotonic()
    frame = read_frame()
    span = time.monotonic() - start
    logger.info(f"get_target_position -> {frame.get('reason')} ({span:.3f} s)")
    logger.debug(f"get_target_position -> {frame}")
    if not frame["ok"]:
        return frame
    for o in frame["objects"]:
        if f"obj_{o['id']}" == object_id:
            x, y, z = o["position_world"]
            return trim({"ok": True, "reason": "ok", "object_id": object_id,
                         "x": x, "y": y, "z": z, "yaw": o["yaw_world"]})
    return {"ok": False, "reason": "not_found",
            "detail": f"camera does not see {object_id!r}, call get_objects for the ids"}


@mcp.tool()
def pick(x: float, y: float, z: float, yaw: float = 0.0) -> dict:
    """Pick up the object whose centre is at (x, y, z).

    Take x, y, z and yaw from get_target_position. The gripper must be empty.
    On failure the arm is back above the object with nothing held.

    Args:
        x, y, z: Object centre, metres, world frame.
        yaw: Object rotation about the vertical axis, rad.

    Returns:
        ok, reason, stage (where it stopped), holding, width_mm.
        reason is one of ok, already_holding, unreachable, grasp_failed,
        grasp_slipped, timeout, busy, error.
    """
    err = not_finite(x=x, y=y, z=z, yaw=yaw)
    if err:
        return err
    return trim(call("pick", args={"x": x, "y": y, "z": z, "yaw": yaw}, timeout=30.0))

@mcp.tool()
def place(x: float, y: float, z: float, yaw: float = 0.0) -> dict:
    """Put the held object down so its centre ends at (x, y, z).

    The gripper must be holding something (see pick).

    Args:
        x, y, z: Target centre of the object, metres, world frame.
        yaw: Rotation about the vertical axis, rad.

    Returns:
        ok, reason, stage (where it stopped).
        reason is one of ok, not_holding, unreachable, grasp_slipped,
        timeout, busy, error.
    """
    err = not_finite(x=x, y=y, z=z, yaw=yaw)
    if err:
        return err
    return trim(call("place", args={"x": x, "y": y, "z": z, "yaw": yaw}, timeout=30.0))

@mcp.tool()
def go_home() -> dict:
    """Move the arm back to its start pose and open the gripper.

    Use to recover after a failure or a timeout. Drops anything held.

    Returns:
        ok, reason, stage, dropped (True if it was holding something).
    """
    return trim(call("go_home", timeout=20.0))

# ----------------------------------------- main
def main() -> None:
    """Create the data directory and start the MCP server on stdio.

    Blocks until the client closes the connection.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("Starting Panda MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()