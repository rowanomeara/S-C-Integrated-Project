





from src.simulation.robot_interface import TurtleBotSim

robot = TurtleBotSim()
print("Start pose:", robot.get_ground_truth_pose())

robot.set_velocity(0.15, -0.1)
for _ in range(100):
    robot.step()

print("End pose:  ", robot.get_ground_truth_pose())
print("Odometry:  ", robot.get_odometry())
print("Min LiDAR: ", robot.get_lidar_scan().min())
print("Cam shape: ", robot.get_camera_frame().shape)


