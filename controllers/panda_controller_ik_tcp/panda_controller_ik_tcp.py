"""panda_controller_ik_tcp controller."""
from controller import Robot, Supervisor
import numpy as np
import ikpy
from ikpy.chain import Chain
from ikpy.link import URDFLink

# ---------------------------------------------------------------- Constants

P_INIT = np.array([0, 0, 0, -1.7708, -1.6, 1.6, 0.79])   # from Panda.proto
TCP_OFFSET = 0.1034          # The offset of the TCP from the last link along the approach axis
ACTIVE_LINKS_MASK = [False, False] + [True] * 7 + [False, False] # 2 base + 7 active joints + gripper + tcp
POS_TOL = 1e-3              # max accepted IK position error, in metres
ANG_TOL = np.deg2rad(1.0)   # max accepted IK orientation error, in radians

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

robot.step(timestep)   # sensor values and Supervisor poses are only valid after the first step

# ---------------------------------------------------------------- Helpers

def wait(seconds):
    """Wait for a given number of seconds."""
    end_time = robot.getTime() + seconds
    while robot.step(timestep) != -1 and robot.getTime() < end_time:
        pass

def gripper(width):
    """Per-finger opening width in meters. 0.0 = closed, 0.04 = open."""
    for finger in fingers:
        finger.setPosition(width)
    wait(1.0)  # wait for the gripper to move

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
        return {"reached": False,
                "reason": "unreachable",
                "target": target.tolist(),
                "ik_pos_err_mm": round(float(pos_err) * 1000, 2),
                "ik_ang_err_deg": round(float(np.rad2deg(ang_err)), 2)}

    # Send and wait
    set_joints(ik)
    reached = wait_until_reached(ik)

    return {"reached": reached,
            "reason": "ok" if reached else "timeout",
            "target": target.tolist()}

# ---------------------------------------------------------------- Demo

cube = np.array(robot.getFromDef('CUBE').getPosition())

# Open the gripper
gripper(0.04)

# Move to the cube
print(move_to(cube + np.array([0, 0, 0.1])))
print(move_to(cube))

# Close the gripper
gripper(0.0)

# Move the robot up by 0.5 meters
target = cube + np.array([0, 0, 0.5])
print(move_to(target))

# Print results
R = np.array(robot.getFromDef('GRIPPER').getOrientation()).reshape(3, 3)
print(robot.getFromDef('GRIPPER').getPosition() + R[:, 2] * TCP_OFFSET, "vs", target)


while robot.step(timestep) != -1:
    pass

# Enter here exit cleanup code.
