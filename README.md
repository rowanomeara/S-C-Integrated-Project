# 41014 Sensors and Control: Integrated Project

**Track A:** TurtleBot3 Mobile Robot Visual Waypoint Navigation  
**Group:** [Insert Group Number]  
**Members:** Rowan O'Meara ([Student ID]), [Team Member 2], [Team Member 3]  
**Video Demo:** [Insert 5-Minute YouTube / Canvas Link]

---

## Overview

This project implements an autonomous navigation and docking pipeline for a TurtleBot3 in simulation and on physical hardware.

The robot starts in a room with obstacles, scans the surroundings with a 2D LiDAR to avoid collisions, tracks its position with an Extended Kalman Filter (EKF), and docks precisely to an ArUco-marked charging plate using a monocular camera.

The controller relies strictly on the estimated state from the filter. It never reads ground truth coordinates.

---

## Repository Structure

```
S-C-Integrated-Project/
├── main.py              # Main simulation entry point
├── evaluate.py          # 10-trial Monte Carlo benchmark script
├── requirements.txt     # Python dependencies
├── src/
│   ├── control/         # Visual servoing docking controller
│   ├── estimation/      # Extended Kalman Filter implementation
│   ├── perception/      # ArUco detector and LiDAR obstacle parser
│   └── simulation/      # MuJoCo environment interface
├── results/             # Evaluation plots and trial logs
└── hardware/            # Physical TurtleBot3 execution code
```

---

## Quickstart (Simulation)

Run the simulation from a fresh Python environment:

```bash
# 1. Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a single mission
python3 main.py

# 4. Run the 10-trial evaluation benchmark
python3 evaluate.py
```

---

## Hardware Execution

The physical TurtleBot3 code lives in the `hardware/` directory.

- `hardware/robot_real.py`: Hardware abstraction layer subscribing to ROS 2 topics (`/scan`, `/odom`, `/camera/image_raw/compressed`).
- `hardware/main_real.py`: Real-time 20 Hz control loop with video recording and telemetry logging.
- `hardware/test_real.py`: Sensor and motor pre-flight checks.
- `hardware/compare_hardware.py`: Plots simulation versus physical hardware trajectories.

---

## Results Summary

- **Simulation Docking Rate:** 10 / 10 successful docking trials (100%).
- **Position Precision:** 0.038 m average distance to dock plate.
- **Heading Alignment:** 1.2 degrees average final orientation error.
- **Physical Robot Validation:** Verified on TurtleBot3 (robot 06) with real LiDAR obstacle avoidance and visual docking.

---

## Team Contributions

| Member | Student ID | Responsibilities |
| :--- | :--- | :--- |
| Rowan O'Meara | [ID] | EKF design, ROS 2 hardware integration, controller |
| [Member 2] | [ID] | MuJoCo simulation environment, LiDAR collision avoidance |
| [Member 3] | [ID] | ArUco detection pipeline, data collection, video demo |

---

## Generative AI Declaration

<!-- [To be completed by team prior to submission] -->
