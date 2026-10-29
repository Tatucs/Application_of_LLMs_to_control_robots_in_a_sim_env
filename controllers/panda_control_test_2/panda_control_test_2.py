# Based on ABB arm
"""Demonstration of inverse kinematics using the "ikpy" Python module."""

import sys
import tempfile
import numpy as np
try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

import math
from controller import Supervisor

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')


# Initialize the Webots Supervisor.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))
pandaChain = Chain.from_urdf_file(filename, active_links_mask=[False, False, True, True, True, True,True, True, True, False, False])

# List all parsed links to verify their index, name, and type
for index, link in enumerate(pandaChain.links):
    print(f"Index {index}: {link.name} (Type: {link.joint_type})")

# Initialize the arm motors and encoders.
motors = []
for link in pandaChain.links:
    if 'panda_joint' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

# Get the arm and target nodes.
target = supervisor.getFromDef('TARGET')
arm = supervisor.getSelf()

# Loop 1: Move the arm hand to the target.
print('Move the yellow and black sphere to move the arm...')
while supervisor.step(timeStep) != -1:
    # Get the absolute postion of the target and the arm base.
    targetPosition = target.getPosition()
    armPosition = arm.getPosition()

    # Compute the position of the target relatively to the arm.
    
    x = targetPosition[0] - armPosition[0]
    y = targetPosition[1] - armPosition[1]
    z = targetPosition[2] - armPosition[2]
    
    print("Target in robot frame:", [x,y,z])
    # First create an empty array of zeros to match the number of parsed links
    initial_position = [0]*len(pandaChain.links)
    
    # Build a lookup map of active motor positions by device name
    sensor_values = {m.getName(): m.getPositionSensor().getValue() for m in motors}

    # Populate the chain positions dynamically without hardcoded index additions
    for idx, link in enumerate(pandaChain.links):
        if link.name in sensor_values:
            raw_val = sensor_values[link.name]
            
            # Fallback in case Webots returns NaN before the sensor fully initializes
            if math.isnan(raw_val):
                raw_val = 0.0

            # Check if ikpy successfully parsed physical bounds for this link
            if link.bounds is not None:
                # Force the sensor value strictly inside the [lower, upper] mathematical bounds
                lower_bound, upper_bound = link.bounds
                clamped_val = max(lower_bound, min(raw_val, upper_bound))
            else:
                clamped_val = raw_val

            initial_position[idx] = clamped_val
    # Execute the inverse kinematics
    ikResults = pandaChain.inverse_kinematics([x, y, z], initial_position=initial_position)

    # Recalculate the inverse kinematics of the arm if necessary.
    position = pandaChain.forward_kinematics(ikResults)
    squared_distance = (position[0, 3] - x)**2 + (position[1, 3] - y)**2 + (position[2, 3] - z)**2
    if math.sqrt(squared_distance) > 0.03:
        ikResults = pandaChain.inverse_kinematics([x, y, z], initial_position=initial_position)

    # Build a lookup map of actual motor devices by name
    motor_dict = {m.getName(): m for m in motors}

    # Actuate the arm motors dynamically based on URDF link names
    for idx, link in enumerate(pandaChain.links):
        if link.name in motor_dict:
            motor_dict[link.name].setPosition(ikResults[idx])