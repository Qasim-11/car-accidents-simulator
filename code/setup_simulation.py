import pybullet as p
import pybullet_data
import time
import math


def run():
    # Connect to GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load ground
    p.loadURDF("plane.urdf")

    # Load two cars at initial positions
    car1 = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2])
    car2 = p.loadURDF("racecar/racecar.urdf", [2, 2, 0.2])

    # === CAR 1 sliders ===
    x1_slider = p.addUserDebugParameter("Car1 X", -5, 5, 0)
    y1_slider = p.addUserDebugParameter("Car1 Y", -5, 5, 0)
    yaw1_slider = p.addUserDebugParameter("Car1 Yaw (deg)", -180, 180, 0)

    # === CAR 2 sliders ===
    x2_slider = p.addUserDebugParameter("Car2 X", -5, 5, 2)
    y2_slider = p.addUserDebugParameter("Car2 Y", -5, 5, 2)
    yaw2_slider = p.addUserDebugParameter("Car2 Yaw (deg)", -180, 180, -90)

    # === Start Button ===
    start_button = p.addUserDebugParameter("Start Simulation", 1, 0, 0)  # dummy toggle

    print("🎮 Adjust both cars using sliders. Toggle 'Start Simulation' to run.")

    simulation_started = False

    while True:
        # Read current slider values
        x1 = p.readUserDebugParameter(x1_slider)
        y1 = p.readUserDebugParameter(y1_slider)
        yaw1_rad = math.radians(p.readUserDebugParameter(yaw1_slider))
        pos1 = [x1, y1, 0.2]
        orn1 = p.getQuaternionFromEuler([0, 0, yaw1_rad])

        x2 = p.readUserDebugParameter(x2_slider)
        y2 = p.readUserDebugParameter(y2_slider)
        yaw2_rad = math.radians(p.readUserDebugParameter(yaw2_slider))
        pos2 = [x2, y2, 0.2]
        orn2 = p.getQuaternionFromEuler([0, 0, yaw2_rad])

        # Update car positions before starting
        if not simulation_started:
            p.resetBasePositionAndOrientation(car1, pos1, orn1)
            p.resetBasePositionAndOrientation(car2, pos2, orn2)

        # Check if the user pressed the start button
        if not simulation_started and p.readUserDebugParameter(start_button) > 0.5:
            print("🚦 Starting simulation!")
            p.disconnect()
            return pos1, yaw1_rad, pos2, yaw2_rad

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    pos1, pos2 = run()
    print("Car 1 Position:", pos1)
    print("Car 2 Position:", pos2)
