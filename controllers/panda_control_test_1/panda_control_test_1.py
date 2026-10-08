"""panda_control_test_1 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Supervisor

# create the Robot instance -> using the Supervisor() instead of the Robot() class to have access to the simulation environment as well, not only the robot itself
robot = Supervisor()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# get information about the robot's joints and motors
joints = [robot.getDevice(f"panda_joint{i}") for i in range(1, 8)]
sensors = [robot.getDevice(f"panda_joint{i}_sensor") for i in range(1, 8)]
for sensor in sensors:
    sensor.enable(timestep)
fingers = [robot.getDevice("panda_finger::left"), robot.getDevice("panda_finger::right")]

# Franka poses
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
LEFT = [0.8, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
DOWN = [0.0, 0.3, 0.0, -2.0, 0.0, 2.3, 0.785]

def wait(seconds):
    """Wait for a given number of seconds."""
    end_time = robot.getTime() + seconds
    while robot.step(timestep) != -1 and robot.getTime() < end_time:
        pass

def move_to_pose(pose, timeout=3.0):
    """Command all joints, step until reached or timeout. Returns max joint error (rad)."""
    for joint, q in zip(joints, pose):
        joint.setPosition(q)
    end_time = robot.getTime() + timeout
    while robot.step(timestep) != -1:
        max_error = max(abs(sensor.getValue() - q) for sensor, q in zip(sensors, pose))
        if max_error < 0.01 or robot.getTime() > end_time:
            return max_error

def gripper(width):
    """Per-finger opening width in meters. 0.0 = closed, 0.04 = open."""
    for finger in fingers:
        finger.setPosition(width)
    wait(1.0)  # wait for the gripper to move

# Get the cube node from the simulation environment
cube = robot.getFromDef("CUBE")
if cube is None:
    print("ERROR: no DEF CUBE node found in the current world file")
else:
    print("Found cube at position:", cube.getPosition())

# Move the robot to a few poses and print the max joint error
for name, pose in [("HOME", HOME), ("LEFT", LEFT), ("DOWN", DOWN), ("HOME", HOME)]:
    max_error = move_to_pose(pose)
    print(f"Moving to {name} pose with max joint error: {max_error:.4f} rad")

# Open, then close the gripper
gripper(0.04)
gripper(0.012)
print("Cube at position:", cube.getPosition() if cube else None)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
