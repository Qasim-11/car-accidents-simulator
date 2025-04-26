import pybullet as p
import time
import pybullet_data
import setup_simulation
import torch
n = input("Enter the number of the simulation: ")
pos1, start_orientation1, pos2, start_orientation2 = torch.load(f"pos1_{n}.pt"), torch.load(f"start_orientation1_{n}.pt"), torch.load(f"pos2_{n}.pt"), torch.load(f"start_orientation2_{n}.pt")
# return pos1, yaw1_rad, pos2, yaw2_rad

time.sleep(1) # wait for 1
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")


boxId1 = p.loadURDF("racecar/racecar.urdf",pos1, list(start_orientation1))
boxId2 = p.loadURDF("racecar/racecar.urdf",pos2, list(start_orientation2))

# Get wheel and steering joints for boxId1
wheel_joints = []
steering_joints = []
for j in range(p.getNumJoints(boxId1)):
    joint_name = p.getJointInfo(boxId1, j)[1]
    if b'wheel' in joint_name:
        wheel_joints.append(j)
    if b'steering' in joint_name:
        steering_joints.append(j)

# Set desired steering angle in radians
# Positive = left, Negative = right
steering_angle_parameter = 0.3  # Turn left (try -0.3 for right turn)

# Set target speed
target_velocity_parameter = 20  # Adjusted to roughly get 2 m/s
force = 10

out = torch.load(f"cars_parameters{n}.pt")



for i in range(240 * 10):
    # Set wheel speeds
    for wheel in wheel_joints:
        p.setJointMotorControl2(
            boxId1,
            wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_velocity_parameter,
            force=out[i, 0, 0].item()
        )         # step, speed, car

    for wheel in wheel_joints:
        p.setJointMotorControl2(
            boxId2,
            wheel,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_velocity_parameter,
            force=out[i, 0, 1].item()
            

        )
    
    # Set steering angle
    for steer in steering_joints:
        p.setJointMotorControl2(
            boxId1,
            steer,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.2,
            force=out[i, 1, 0].item()
                  # step, steering, car    

        )

    # Set steering angle
    for steer in steering_joints:
        p.setJointMotorControl2(
            boxId2,
            steer,
            controlMode=p.POSITION_CONTROL,
            targetPosition=-steering_angle_parameter,
            force=out[i, 1, 1].item()
        )


    p.stepSimulation()
    time.sleep(1.0 / 240.0)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId1)
p.disconnect()

 