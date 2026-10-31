"""panda_mcp_controller controller."""
from controller import Robot, Supervisor
import numpy as np
import json
import os
import time
from pathlib import Path
import ikpy
from ikpy.chain import Chain
from ikpy.link import URDFLink

# ---------------------------------------------------------------- Constants

P_INIT = np.array([0, 0, 0, -1.7708, -1.6, 1.6, 0.79])   # from Panda.proto
TCP_OFFSET = 0.1034         # The offset of the TCP from the last link along the approach axis
ACTIVE_LINKS_MASK = [False, False] + [True] * 7 + [False, False] # 2 base + 7 active joints + gripper + tcp
POS_TOL = 1e-3              # max accepted IK position error, in metres
ANG_TOL = np.deg2rad(1.0)   # max accepted IK orientation error, in radians

FINGER_OPEN = 0.04          # per-finger max opening, in metres (pad gap = left + right)
EMPTY_GAP = 3e-3            # pad gap below this = closed on nothing, in metres
OPEN_TOL = 0.5e-3           # per-finger tolerance for "fully open", in metres
STOP_TOL = 1e-4             # gap change below this over STOP_STEPS steps = fingers stopped
STOP_STEPS = 5
GRIPPER_TIMEOUT = 2.0       # seconds

APPROACH_HEIGHT = 0.10      # TCP clearance above the object before descending / after lifting, in metres
RELEASE_HEIGHT = 0.005      # place releases this far above the target, so the object is not pressed into the table

HEARTBEAT_PERIOD = 0.5      # s sim time between idle status writes
OBJECT_PREFIXES = ["CUBE","BASKET"]      # DEF prefix of graspable objects 
COLOURS = {"red": (0.666667, 0, 0), "green": (0, 0.333333, 0), "blue": (0,0, 0.498039),
           "yellow": (1, 1, 0), "pink": (1, 0.333333, 1), "brown": (0.439216,0.258824,0.0784314)}
OBJECT_KINDS ={"cube":"graspable", "basket":"area"}
CUBE_SIZE=0.04

# ---------------------------------------------------------------- Setup

# create the Robot instance.
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Get and inspect the robot's URDF
urdf = robot.getUrdf()
with open("panda.urdf", "w") as f:
    f.write(urdf)

print("The robots base is at: ", robot.getSelf().getPosition())
print("The robots base orientation is: ", robot.getSelf().getOrientation())

# Create the kinematic chain from the URDF file
full_chain = Chain.from_urdf_file("panda.urdf", active_links_mask=ACTIVE_LINKS_MASK)

# Tool center point
tcp = URDFLink(name="tcp",
               origin_translation=[0, 0, TCP_OFFSET],
               origin_orientation=[0, 0, 0],
               rotation=None,
               joint_type="fixed")
chain = Chain(full_chain.links[:10] + [tcp], active_links_mask=ACTIVE_LINKS_MASK)

# Shift the bounds to ikpy coordinates
for link, p in zip(chain.links[2:9], P_INIT):
    lo, hi = link.bounds
    link.bounds = (lo - p, hi - p)

# Devices
link_names = [link.name for link in chain.links]
motor_names = link_names[2:9]
motors = [robot.getDevice(motor_name) for motor_name in motor_names]
fingers = [robot.getDevice("panda_finger::left"), robot.getDevice("panda_finger::right")]

# Enable sensors
sensors = [robot.getDevice(f"{motor_name}_sensor") for motor_name in motor_names]
for sensor in sensors:
    sensor.enable(timestep)

finger_sensors = [robot.getDevice("panda_finger::left_sensor"), robot.getDevice("panda_finger::right_sensor")]
for sensor in finger_sensors:
    sensor.enable(timestep)
for finger in fingers:
    finger.enableForceFeedback(timestep)

robot.step(timestep)   # sensor values and Supervisor poses are only valid after the first step

# ---------------------------------------------------------------- File I/O

# <repo>/data/panda. The MCP server resolves <repo>/data from its own __file__, so both sides meet here.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "panda"
COMMANDS_FILE = DATA_DIR / "commands.json"
STATUS_FILE = DATA_DIR / "status.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("Data dir:", DATA_DIR)

def write_json(path, data, retries=5):
    """Writes to a .tmp file, then swap it in, so the reader never sees
    a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:      # Windows: the other process has the file open at this instant
            time.sleep(0.01)
    os.replace(tmp, path)            # last attempt: a real failure should be visible -> raise

def read_json(path):
    """Parsed JSON, or None if the file is missing, half-written or locked.
    None means 'not there yet' so the read should be retried later. Does not raise exceptions."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None

# ---------------------------------------------------------------- Helpers

def wait(seconds):
    """Wait for a given number of seconds."""
    end_time = robot.getTime() + seconds
    while robot.step(timestep) != -1 and robot.getTime() < end_time:
        pass

def wait_until_reached(ik, tol=1e-3, timeout_ms=5000):
    """Wait until the robot reaches the commanded joint angles (or timeout)."""
    for _ in range(int(timeout_ms / timestep)):
        robot.step(timestep)
        q_now = np.array([s.getValue() for s in sensors])
        if np.max(np.abs(q_now - (ik[2:9] + P_INIT))) < tol:
            return True
    return False                             # timed out

def read_q():
    """Current joint angles as an ikpy vector (sensors - P_INIT, padded for the fixed links)."""
    return np.concatenate(([0, 0], [s.getValue() for s in sensors] - P_INIT, [0, 0]))

def set_joints(ik):
    """Send an ikpy vector to the motors (+ P_INIT back to Webots coordinates)."""
    for motor, angle in zip(motors, ik[2:9] + P_INIT):
        motor.setPosition(angle)

def world_to_base(p_world, R_world):
    """Transform a world-frame position and orientation into the robot base frame
    (the frame the ikpy chain works in). Base pose is read each call, so it
    stays correct if the robot is moved."""
    t_base = np.array(robot.getSelf().getPosition())
    R_base = np.array(robot.getSelf().getOrientation()).reshape(3, 3)
    p_base = R_base.T @ (np.array(p_world) - t_base)
    R_in_base = R_base.T @ R_world
    return p_base, R_in_base

def base_to_world(p_base):
    """Inverse of world_to_base for a position: base frame -> world frame."""
    t_base = np.array(robot.getSelf().getPosition())
    R_base = np.array(robot.getSelf().getOrientation()).reshape(3, 3)
    return R_base @ p_base + t_base

def tcp_pos():
    """TCP position in the world frame, from the joint sensors."""
    return base_to_world(chain.forward_kinematics(read_q())[:3, 3])

def pose_error(ik, p_target, R_target):
    """FK of an ikpy vector vs a target pose (base frame).
    Returns (position error in metres, orientation error in radians)."""
    T = chain.forward_kinematics(ik)
    pos_err = np.linalg.norm(T[:3, 3] - p_target)
    R_err = T[:3, :3].T @ R_target               # rotation still missing from the solution
    ang_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    return pos_err, ang_err

# ---------------------------------------------------------------- move_to

def move_to(target_world, yaw=0.0):
    """Move the TCP to target_world (x, y, z in metres, world frame),
    gripper pointing down, rotated by yaw (rad) about the world z axis.
    Returns a result dict."""
    target = np.array(target_world)

    # Gripper pointing down, rotated by yaw around the world z axis
    z = np.array([0, 0, -1])
    x = np.array([np.cos(yaw), np.sin(yaw), 0])
    y = np.cross(z, x)
    R_target = np.column_stack((x, y, z))

    # World -> base frame (both the position and the orientation)
    target_base, R_target_base = world_to_base(target, R_target)

    # Solve IK, seeded from the current pose
    ik = chain.inverse_kinematics(target_base,
                                  orientation_mode="all",
                                  target_orientation=R_target_base,
                                  initial_position=read_q())

    # Verify the IK solution with FK before moving (ikpy never reports failure itself)
    pos_err, ang_err = pose_error(ik, target_base, R_target_base)
    if pos_err > POS_TOL or ang_err > ANG_TOL:
        return {"ok": False,
                "reason": "unreachable",
                "target": np.round(target, 4).tolist(),
                "ik_pos_err_mm": round(float(pos_err) * 1000, 2),
                "ik_ang_err_deg": round(float(np.rad2deg(ang_err)), 2)}

    # Send and wait
    set_joints(ik)
    reached = wait_until_reached(ik)

    # Measure where the TCP actually is: FK of the joint sensors (encoders)
    q = read_q()
    tcp_world = base_to_world(chain.forward_kinematics(q)[:3, 3])
    pos_err, ang_err = pose_error(q, target_base, R_target_base)

    return {"ok": reached,
            "reason": "ok" if reached else "timeout",
            "target": np.round(target, 4).tolist(),
            "reached": np.round(tcp_world, 4).tolist(),
            "pos_err_mm": round(float(pos_err) * 1000, 2),
            "ang_err_deg": round(float(np.rad2deg(ang_err)), 2)}

# ---------------------------------------------------------------- Gripper

def finger_width():
    """Gap between the finger pads in metres (sum of the two finger positions)."""
    return sum(s.getValue() for s in finger_sensors)

def gripper_state():
    """Read-only gripper state, no motion. Holding = the pads stopped
    between closed-on-nothing and fully open."""
    left, right = (s.getValue() for s in finger_sensors)
    width = left + right
    holding = EMPTY_GAP < width < 2 * (FINGER_OPEN - OPEN_TOL)
    return {"holding": bool(holding),
            "width_mm": round(width * 1000, 1),
            "left_mm": round(left * 1000, 1),
            "right_mm": round(right * 1000, 1),
            "force_n": [round(f.getForceFeedback(), 1) for f in fingers]}

def close_gripper(timeout=GRIPPER_TIMEOUT):
    """Close until the fingers stop (on an object or fully closed).
    Closing on nothing is a failed grasp. Returns a result dict."""
    for finger in fingers:
        finger.setPosition(0.0)

    # Stopped = the gap changed less than STOP_TOL over the last STOP_STEPS steps.
    # A target-reached check would never fire: with a cube in between, the fingers never get to 0.
    start = robot.getTime()
    widths = []
    stopped = False
    while robot.getTime() - start < timeout and robot.step(timestep) != -1:
        widths.append(finger_width())
        if len(widths) > STOP_STEPS and abs(widths[-1] - widths[-1 - STOP_STEPS]) < STOP_TOL:
            stopped = True
            break

    state = gripper_state()
    if not stopped:
        reason = "timeout"
    elif not state["holding"]:
        reason = "grasp_failed"
    else:
        reason = "ok"
    return {"ok": reason == "ok", "reason": reason, **state,
            "time_s": round(robot.getTime() - start, 3)}

def open_gripper(timeout=GRIPPER_TIMEOUT):
    """Open fully. A timeout means the fingers were blocked from outside,
    so it is reported as a failure. Returns a result dict."""
    for finger in fingers:
        finger.setPosition(FINGER_OPEN)

    start = robot.getTime()
    opened = False
    while robot.getTime() - start < timeout and robot.step(timestep) != -1:
        if all(abs(s.getValue() - FINGER_OPEN) < OPEN_TOL for s in finger_sensors):
            opened = True
            break

    return {"ok": opened, "reason": "ok" if opened else "timeout", **gripper_state(),
            "time_s": round(robot.getTime() - start, 3)}

# ---------------------------------------------------------------- pick / place

def pick(xyz, yaw=0.0):
    """Grasp the object whose centre is at xyz (x, y, z in metres, world frame),
    gripper rotated by yaw (rad). A failure after the approach leaves the arm
    above xyz with the gripper open. Returns a result dict; stage = where it stopped.
    """
    xyz = np.array(xyz)
    above = xyz + [0, 0, APPROACH_HEIGHT]

    # Never grasp a second object with the first one in hand
    if gripper_state()["holding"]:
        return {"ok": False, "reason": "already_holding", "stage": "precheck", **gripper_state()}

    result = open_gripper()
    if not result["ok"]:
        return {**result, "stage": "open"}

    result = move_to(above, yaw)
    if not result["ok"]:
        return {**result, "stage": "approach"}

    result = move_to(xyz, yaw)
    if not result["ok"]:
        move_to(above, yaw)                  # back up: the next call must not start down at the table
        return {**result, "stage": "descend"}

    result = close_gripper()
    if not result["ok"]:
        open_gripper()
        move_to(above, yaw)
        return {**result, "stage": "close"}

    result = move_to(above, yaw)
    if not result["ok"]:
        return {**result, "stage": "lift"}

    # The object can slip out while lifting
    state = gripper_state()
    if not state["holding"]:
        return {"ok": False, "reason": "grasp_slipped", "stage": "lift", **state}

    return {"ok": True, "reason": "ok", "stage": "done", **state}

def place(xyz, yaw=0.0):
    """Put the held object so its centre ends at xyz (x, y, z in metres, world frame),
    gripper rotated by yaw (rad). Releases RELEASE_HEIGHT above xyz.
    Returns a result dict; stage = where it stopped."""
    xyz = np.array(xyz)
    above = xyz + [0, 0, APPROACH_HEIGHT]

    # Nothing to place: return without moving
    if not gripper_state()["holding"]:
        return {"ok": False, "reason": "not_holding", "stage": "precheck", **gripper_state()}

    result = move_to(above, yaw)
    if not result["ok"]:
        return {**result, "stage": "approach"}

    result = move_to(xyz + [0, 0, RELEASE_HEIGHT], yaw)
    if not result["ok"]:
        move_to(above, yaw)
        return {**result, "stage": "descend"}

    # The object may have slipped during the transfer
    state = gripper_state()
    if not state["holding"]:
        move_to(above, yaw)
        return {"ok": False, "reason": "grasp_slipped", "stage": "descend", **state}

    result = open_gripper()
    if not result["ok"]:
        return {**result, "stage": "release"}

    result = move_to(above, yaw)
    if not result["ok"]:
        return {**result, "stage": "retract"}

    return {"ok": True, "reason": "ok", "stage": "done"}

def go_home():
    """Joint-space move back to the startup pose P_INIT (hand down, yaw -45 deg),
    then open the gripper.
    Always opens: dropped = True if it was holding something.
    Returns a result dict; stage = where it stopped."""
    was_holding = gripper_state()["holding"]

    ik = np.zeros(len(chain.links))          # ikpy zero = P_INIT
    set_joints(ik)
    if not wait_until_reached(ik):
        return {"ok": False, "reason": "timeout", "stage": "move", "dropped": False}

    result = open_gripper()
    if not result["ok"]:
        return {**result, "stage": "open", "dropped": False}

    return {"ok": True, "reason": "ok", "stage": "done", "dropped": was_holding,
            "reached": np.round(tcp_pos(), 4).tolist()}

# ---------------------------------------------------------------- Perception tools

def scene_nodes(prefix:list[str]):
    """Top-level scene nodes whose DEF starts with prefix."""
    children = robot.getRoot().getField("children")
    nodes = (children.getMFNode(i) for i in range(children.getCount()))
    return [n for n in nodes if n.getDef().startswith(tuple(prefix))]

def node_colour(node):
    """Nearest colour word to the node's PRBAppearance baseColor
    (until camera recognitionColors)."""
    try:
        rgb = np.array(node.getField("PBRAppearance").getSFNode().getField("baseColor").getSFColor())
    except AttributeError:      # getField / getSFNode returned None: no appearance or no baseColor
        return "unknown"
    return min(COLOURS, key=lambda c: np.linalg.norm(rgb - np.array(COLOURS[c])))

def get_objects():
    """All graspable and non graspable objects: id + kind + colour."""
    objects = [{"id": n.getDef().lower(), "kind": OBJECT_KINDS.get(n.getField("model").getSFString(),"unknown"), "colour": node_colour(n)} for n in scene_nodes(OBJECT_PREFIXES)]
    return {"ok": True, "reason": "ok", "objects": objects}

def get_target_position(object_id):
    """World-frame centre (m) and yaw (rad) of one GRASPABLE object, ready to pass to pick."""
    node = robot.getFromDef(str(object_id).upper())
    if node is None or not node.getDef().startswith(OBJECT_PREFIXES) or OBJECT_KINDS.get(node.getField("model").getSFString())!="graspable":
        return {"ok": False, "reason": "not_found",
                "detail": f"no object {object_id!r}, call get_objects for the ids"}
    x, y, z = node.getPosition()
    R = np.array(node.getOrientation()).reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))     # rotation about world z
    return {"ok": True, "reason": "ok", "object_id": object_id,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "yaw": round(yaw, 3)}

def get_area_position(object_id):
    """World_frame centre (m), shape and x and y extreme values of an area object, ready to pass to place. The edges of the area are aligned with word coordinate system."""
    node = robot.getFromDef(str(object_id).upper())
    if node is None or not node.getDef().startswith(OBJECT_PREFIXES) or OBJECT_KINDS.get(node.getField("model").getSFString())!="area":
        return {"ok": False, "reason": "not_found",
                "detail": f"no object {object_id!r}, call get_objects for the ids"}
    x, y, z = node.getPosition()
    shape=node.getField("children").getMFNode(0)
    area= shape.getField("geometry").getSFNode()
    width, depth,_ = area.getField("size").getSFVec3f()
    return {"ok": True, "reason": "ok", "object_id": object_id,"shape":"rectangle", 
            "center_x": round(x, 3), "center_y": round(y, 3), "center_z": round(z, 3), "x_min":x-(width/2-CUBE_SIZE/2), "x_max":x+(width/2-CUBE_SIZE/2), "y_min":y-(depth/2-CUBE_SIZE/2), "y_max":y+(depth/2-CUBE_SIZE/2)}

def get_gripper_state():
    """Read-only gripper state, ready to return to the client."""
    state = gripper_state()
    return {"ok": True, "reason": "ok", **state}

# ---------------------------------------------------------------- Command loop

def write_status(**changes):
    """Merges the changes into status, stamps the heartbeat, writes status.json."""
    status.update(changes)
    status["holding"] = gripper_state()["holding"]
    status["heartbeat"] = {"sim_time": round(robot.getTime(), 3), "wall_time": time.time()}
    write_json(STATUS_FILE, status)

def dispatch(cmd):
    """Runs one command and returns a result dict. Never raises."""
    func = ACTIONS.get(cmd.get("action"))
    if func is None:                                # unknown action
        return {"ok": False, "reason": "not_found", "detail": f"unknown action {cmd.get('action')!r}"}
    try:
        return func(**(cmd.get("args") or {}))
    except Exception as e:
        return {"ok": False, "reason": "error", "detail": f"{type(e).__name__}: {e}"}

def wrap_yaw(yaw):
    """Wraps yaw into [-pi/4, pi/4]. A cube can be grasped at any multiple of
    90 deg from its own yaw; the one closest to 0 keeps the wrist away from its limits."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

def cmd_pick(x, y, z, yaw=0.0):
    return pick([x, y, z], wrap_yaw(yaw))

def cmd_place(x, y, z, yaw=0.0):
    return place([x, y, z], wrap_yaw(yaw))


# ---------------------------------------------------------------- Main loop

status = {"heartbeat": None, "busy_with": None, "last_result": None, "holding": False}
ACTIONS = {"pick": cmd_pick,
           "place": cmd_place,
           "go_home": go_home,
           "get_objects": get_objects,
           "get_target_position": get_target_position,
           "get_gripper_state": get_gripper_state}   # action name -> function

cmd = read_json(COMMANDS_FILE)
last_id = cmd.get("id") if isinstance(cmd, dict) else None   # stale command
print("Ignoring stale command id:", last_id)
write_status()
last_beat = robot.getTime()

while robot.step(timestep) != -1:
    cmd = read_json(COMMANDS_FILE)
    if isinstance(cmd, dict) and cmd.get("id") is not None and cmd["id"] != last_id:
        last_id = cmd["id"]
        write_status(busy_with=last_id)
        result = dispatch(cmd)
        write_status(busy_with=None, last_result={**result, "id": last_id})
        last_beat = robot.getTime()
    elif robot.getTime() - last_beat >= HEARTBEAT_PERIOD:
        write_status()
        last_beat = robot.getTime()
 