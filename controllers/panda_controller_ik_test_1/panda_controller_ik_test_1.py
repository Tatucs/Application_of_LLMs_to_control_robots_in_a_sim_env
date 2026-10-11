"""panda_controller_ik_test_1 controller."""

from controller import Robot, Supervisor
import numpy as np
import ikpy
from ikpy.chain import Chain

# create the Robot instance.
robot = Supervisor()
urdf = robot.getUrdf()
with open("panda.urdf", "w") as f:
    f.write(urdf)

print("The robots base is at: ", robot.getSelf().getPosition())
print("The robots base orientation is: ", robot.getSelf().getOrientation())

# Inspect the robot's URDF
full_chain = Chain.from_urdf_file("panda.urdf",
              active_links_mask=[False, False] + [True] * 7 + [False, False])
chain = Chain(full_chain.links[:10],
              active_links_mask=[False, False] + [True] * 7 + [False])
link_names = [link.name for link in chain.links]

print(f"The robot has {len(chain.links)} links.")
print(f"The robot has {sum(chain.active_links_mask)} active links.")
print(f"The link names are: {link_names}")

# Get the motor names for the active links
motor_names = link_names[2:9]
joint_bounds = [link.bounds for link in chain.links[2:9]]
print("The active motors with their respective bounds are:")
for name, bounds in zip(motor_names, joint_bounds):
    print(f"{name}: {bounds}")

# List of joint angles
# The initial joint angles are set to the values from the Panda.proto file.
# The robot.getUrdf() method returns the URDF of the robot, which contains the initial joint angles
# and does not substract the initial joint angles from the Panda.proto file.
# Therefore, we need to substract the initial joint angles from the Panda.proto file to get the correct joint angles for the robot.

P_INIT = np.array([0, 0, 0, -1.7708, -1.6, 1.6, 0.79])   # from Panda.proto
q_webots = np.array([0, 0, 0, -1.5, 0, 3, 0])            # The desired joint angles
q = np.concatenate(([0, 0], q_webots - P_INIT, [0]))     # concatanate

# The URDF joint limits are still in Webots coordinates, shift them into URDF coordinates too
for link, p in zip(chain.links[2:9], P_INIT):
    lo, hi = link.bounds
    link.bounds = (lo - p, hi - p)
print("The joint bounds in URDF coordinates are:")
for link in chain.links[2:9]:
    print(f"{link.name}: {float(link.bounds[0]):.2f} to {float(link.bounds[1]):.2f}")


mx_forward = chain.forward_kinematics(q)
print(f"The forward kinematics of the robot is: \n {mx_forward}")

# Set motor positions to the joint angles
for motor_name, angle in zip(motor_names, q_webots):
    motor = robot.getDevice(motor_name)
    motor.setPosition(angle)
    print(f"Set motor {motor_name} to position {angle}")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Enable the joint position sensors to track actual positions of the motors.
sensors = [robot.getDevice(f"{motor_name}_sensor") for motor_name in motor_names]
for sensor in sensors:
    sensor.enable(timestep)

# Wait for the robot to reach the target positions
steps = int(4000 / timestep)   # 4000 ms of simulated time
for _ in range(steps):
    robot.step(timestep)

# Postition comparison
for motor_name, sensor in zip(motor_names, sensors):
    print(f"Motor {motor_name} is at position {sensor.getValue()}")
print(f"The position of the end effector is: {mx_forward[:3, 3]}")
print(f"The position of the end effector from Webots is: {robot.getFromDef('GRIPPER').getPosition()}")

print(f"The orientation of the end effector is: \n {mx_forward[:3, :3]}")
print(f"The orientation of the end effector from Webots is: \n {robot.getFromDef('GRIPPER').getOrientation()}")

# --- Inverse Kinematics ---

# Determining target position with forward kinematics
q_webots_ik = np.array([0, 0.5, 0, -1.5, 0, 3, 1]) # totally random
q_ik_concat = np.concatenate(([0, 0], q_webots_ik - P_INIT, [0]))
mx_forward_ik = chain.forward_kinematics(q_ik_concat)
target_ik = mx_forward_ik[:3,3]

# Calculate inverse kinematics for the target position from the initial joint angles
q_ik_initial = np.concatenate(([0, 0], [s.getValue() for s in sensors] - P_INIT, [0]))
ik_result = chain.inverse_kinematics(target_ik, initial_position=q_ik_initial)

# Compare the result with the target position
error = np.linalg.norm(chain.forward_kinematics(ik_result)[:3, 3] - target_ik)
q_cmd = ik_result[2:9] + P_INIT


# Set motor positions to the inverse kinematics result
for motor_name, angle in zip(motor_names, q_cmd):
    motor = robot.getDevice(motor_name)
    motor.setPosition(angle)
    print(f"Set motor {motor_name} to position {angle}")

# Wait for the robot to reach the target positions
steps = int(4000 / timestep)   # 4000 ms of simulated time
for _ in range(steps):
    robot.step(timestep)

# Print the results
print(f"The target position for inverse kinematics is: {target_ik}")
print(f"The inverse kinematics result is: {ik_result}")
print(f"The error between the target position and the result is: {error}")
print(f"The position of the end effector from Webots is: {robot.getFromDef('GRIPPER').getPosition()}")


while robot.step(timestep) != -1:
    pass

# Enter here exit cleanup code.
