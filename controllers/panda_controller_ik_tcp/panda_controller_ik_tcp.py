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

# ---------------------------------------------------------------- Demo

# Define the target position and orientation for the end effector
target = robot.getFromDef('CUBE').getPosition()
yaw = 0.0   # rotation around z axis
z = np.array([0, 0, -1])
x = np.array([np.cos(yaw), np.sin(yaw), 0])
y = np.cross(z, x)
R_target = np.column_stack((x, y, z))

ik = chain.inverse_kinematics(target,
                              orientation_mode="all",
                              target_orientation=R_target,
                              initial_position=read_q())

# Open the gripper
gripper(0.04)

# Set motor positions to the inverse kinematics result
set_joints(ik)
wait_until_reached(ik)

# Close the gripper
gripper(0.0)

# Move the robot up by 0.5 meters
target = target + np.array([0, 0, 0.5])   # move up
q_now = ik
ik = chain.inverse_kinematics(target,
                              orientation_mode="all",
                              target_orientation=R_target,
                              initial_position=q_now)
set_joints(ik)
wait_until_reached(ik)

# Print results
R = np.array(robot.getFromDef('GRIPPER').getOrientation()).reshape(3, 3)
print(robot.getFromDef('GRIPPER').getPosition() + R[:, 2] * TCP_OFFSET, "vs", target)


while robot.step(timestep) != -1:
    pass

# Enter here exit cleanup code.
