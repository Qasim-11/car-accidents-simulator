import pybullet as p
import time
import pybullet_data
import torch
import setup_simulation
n = input("Enter the number of the simulation: ")
pos1, yaw1_rad, pos2, yaw2_rad = setup_simulation.run()
# return pos1, yaw1_rad, pos2, yaw2_rad

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)  # ← This disables keyboard control
p.setGravity(0, 0, -10)

# Load plane and racecars
planeId = p.loadURDF("plane.urdf")

start_orientation1 = p.getQuaternionFromEuler([0, 0, yaw1_rad])
start_orientation2 = p.getQuaternionFromEuler([0, 0, yaw2_rad])
box_id1 = p.loadURDF("racecar/racecar.urdf", pos1, start_orientation1)
box_id2 = p.loadURDF("racecar/racecar.urdf", pos2, start_orientation2)

cars_parameters = torch.zeros((240 * 10, 2, 2))


# Allow cars to settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

# Get joints for both cars
def get_joints(car_id):
    steering_joints = []
    rear_wheel_joints = []
    for j in range(p.getNumJoints(car_id)):
        joint_name = p.getJointInfo(car_id, j)[1]
        if b"steering" in joint_name:
            steering_joints.append(j)
        if b"wheel" in joint_name and b"rear" in joint_name:
            rear_wheel_joints.append(j)
    return steering_joints, rear_wheel_joints

steering_joints1, rear_wheels1 = get_joints(box_id1)
steering_joints2, rear_wheels2 = get_joints(box_id2)

# Disable default motor control
for j in range(p.getNumJoints(box_id1)):
    p.setJointMotorControl2(box_id1, j, controlMode=p.VELOCITY_CONTROL, force=0)
for j in range(p.getNumJoints(box_id2)):
    p.setJointMotorControl2(box_id2, j, controlMode=p.VELOCITY_CONTROL, force=0)

# Parameters
velocity_limit1 = 20
velocity_limit2 = 20
max_steering = 0.5

# Fix camera position and angle
p.resetDebugVisualizerCamera(
    cameraDistance=5,
    cameraYaw=50,
    cameraPitch=-30,
    cameraTargetPosition=[0, 1, 0.2]
)


for i in range(240 * 10):  # Simulate for 5 seconds at 240 FPS
    keys = p.getKeyboardEvents()
    velocity1 = 0
    steering1 = 0
    velocity2 = 0
    steering2 = 0

    # Car 1 controls
    if ord('+') in keys and keys[ord('+')] & p.KEY_WAS_TRIGGERED:
        velocity_limit1 += 2
        print(f"Car 1 Speed limit increased to {velocity_limit1}")
    if ord('-') in keys and keys[ord('-')] & p.KEY_WAS_TRIGGERED:
        velocity_limit1 = max(0, velocity_limit1 - 2)
        print(f"Car 1 Speed limit decreased to {velocity_limit1}")
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        velocity1 = velocity_limit1
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        velocity1 = -velocity_limit1
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        steering1 = max_steering
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        steering1 = -max_steering

    # Car 2 controls
    if ord('[') in keys and keys[ord('[')] & p.KEY_WAS_TRIGGERED:
        velocity_limit2 += 2
        print(f"Car 2 Speed limit increased to {velocity_limit2}")
    if ord(']') in keys and keys[ord(']')] & p.KEY_WAS_TRIGGERED:
        velocity_limit2 = max(0, velocity_limit2 - 2)
        print(f"Car 2 Speed limit decreased to {velocity_limit2}")
    if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
        velocity2 = velocity_limit2
    if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
        velocity2 = -velocity_limit2
    if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
        steering2 = max_steering
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        steering2 = -max_steering

    # Apply controls to Car 1
    for wheel in rear_wheels1:
        p.setJointMotorControl2(box_id1, wheel, p.VELOCITY_CONTROL, targetVelocity=velocity1, force=10)
    for steer in steering_joints1:
        p.setJointMotorControl2(box_id1, steer, p.POSITION_CONTROL, targetPosition=steering1, force=5)

    # Apply controls to Car 2
    for wheel in rear_wheels2:
        p.setJointMotorControl2(box_id2, wheel, p.VELOCITY_CONTROL, targetVelocity=velocity2, force=10)
    for steer in steering_joints2:
        p.setJointMotorControl2(box_id2, steer, p.POSITION_CONTROL, targetPosition=steering2, force=5)

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    cars_parameters[i, 0, 0] = velocity1
    cars_parameters[i, 0, 1] = velocity2
    cars_parameters[i, 1, 0] = steering1
    cars_parameters[i, 1, 1] = steering2
    # Print cars_parameters values for debugging



torch.save(cars_parameters, f'cars_parameters{n}.pt')
# Save pos1, start_orientation1 locally:
torch.save(pos1, f'pos1_{n}.pt')
torch.save(start_orientation1, f'start_orientation1_{n}.pt')

torch.save(pos2, f'pos2_{n}.pt')
torch.save(start_orientation2, f'start_orientation2_{n}.pt')
