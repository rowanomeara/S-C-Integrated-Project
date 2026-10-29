import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from src.simulation.robot_interface import TurtleBotSim
from src.perception.aruco_detector import ArucoDetector
from src.perception.lidar import LidarProcessor
from src.estimation.ekf import RobotEKF, wrap
from src.control import DockingController

COLS = ["t", "x_true", "y_true", "th_true", "x_est", "y_est", "th_est",
        "x_odom", "y_odom", "th_odom", "sigma_x", "sigma_y", "sigma_th",
        "err_x", "err_y", "err_th", "v_cmd", "w_cmd", "marker_seen"]


def draw_arena(ax):
    """Draws charging dock, ArUco marker, and arena obstacles."""
    ax.add_patch(plt.Rectangle((-1.65, -0.2), 0.15, 0.4, color="gray", alpha=0.5, label="Dock"))
    ax.plot(-1.69, 0, "ks", ms=8, label="ArUco")
    ax.add_patch(plt.Rectangle((-0.35, -0.28), 0.3, 0.56, color="orange", alpha=0.4, label="Obstacle"))
    ax.add_patch(plt.Circle((0.45, 0.65), 0.15, color="teal", alpha=0.3))
    ax.add_patch(plt.Circle((0.45, -0.65), 0.15, color="teal", alpha=0.3))
    ax.set_xlim(-1.9, 1.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.5)


def run_single_trial(robot, aruco, lidar, ekf, x0, max_steps=4200):
    """Executes a single headless simulation trial, returning trajectory logs and summary metrics."""
    robot.reset()
    robot.data.qpos[:3] = [x0[0], x0[1], 0.033]
    robot.data.qpos[3:7] = [np.cos(x0[2] / 2.0), 0, 0, np.sin(x0[2] / 2.0)]
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)

    ekf.x = x0.astype(float).copy()
    ekf.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])
    x_odom = x0.astype(float).copy()
    target_dock = np.array([-1.52, 0.0])
    marker_pos = np.array([-1.69, 0.0])
    controller = DockingController(target_dock=target_dock)

    records, docked = [], False

    for step in range(max_steps):
        t = step * robot.dt
        true_pose = robot.get_ground_truth_pose()
        v_odom, w_odom = robot.get_odometry()

        x_odom[:2] += [v_odom * np.cos(x_odom[2]) * robot.dt, v_odom * np.sin(x_odom[2]) * robot.dt]
        x_odom[2] = wrap(x_odom[2] + w_odom * robot.dt)
        ekf.predict(v_odom, w_odom, dt=robot.dt)

        frame = robot.get_camera_frame() if step % 2 == 0 else None
        r, phi = aruco.detect(frame, annotate=False) if frame is not None else (None, None)
        if r is not None:
            ekf.update_aruco(r, phi, dock_pos=marker_pos)

        ranges = robot.get_lidar_scan()
        d_front, _ = lidar.get_front_clearance(ranges)
        d_left = float(np.mean(ranges[195:255]))
        d_right = float(np.mean(ranges[105:165]))

        dist_to_dock = np.hypot(target_dock[0] - ekf.x[0], target_dock[1] - ekf.x[1])
        if dist_to_dock < 0.18 or (r is not None and r < 0.25) or (dist_to_dock < 0.35 and d_front < 0.25):
            docked = True
            records.append([t, *true_pose, *ekf.x, *x_odom,
                            np.sqrt(ekf.P[0, 0]), np.sqrt(ekf.P[1, 1]), np.sqrt(ekf.P[2, 2]),
                            ekf.x[0] - true_pose[0], ekf.x[1] - true_pose[1], wrap(ekf.x[2] - true_pose[2]),
                            0.0, 0.0, 1 if r is not None else 0])
            break

        cmd_v, cmd_w, _ = controller.compute(ekf.x, r, phi, d_front, d_left, d_right)

        robot.set_velocity(cmd_v, cmd_w)
        robot.step()
        records.append([t, *true_pose, *ekf.x, *x_odom,
                        np.sqrt(ekf.P[0, 0]), np.sqrt(ekf.P[1, 1]), np.sqrt(ekf.P[2, 2]),
                        ekf.x[0] - true_pose[0], ekf.x[1] - true_pose[1], wrap(ekf.x[2] - true_pose[2]),
                        cmd_v, cmd_w, 1 if r is not None else 0])

    arr = np.array(records)
    log = {col: arr[:, i] for i, col in enumerate(COLS)}
    metrics = {
        "docked": docked,
        "duration": float(log["t"][-1]),
        "rmse_pos": float(np.sqrt(np.mean(log["err_x"]**2 + log["err_y"]**2))),
        "rmse_th": float(np.sqrt(np.mean(np.rad2deg(log["err_th"])**2))),
        "dock_err_dist": float(np.hypot(target_dock[0] - log["x_true"][-1], target_dock[1] - log["y_true"][-1])),
        "dock_err_head": float(np.rad2deg(abs(wrap(np.pi - log["th_true"][-1])))),
        "in_2sigma_x": float(np.mean(np.abs(log["err_x"]) <= 2 * log["sigma_x"]) * 100),
        "in_2sigma_y": float(np.mean(np.abs(log["err_y"]) <= 2 * log["sigma_y"]) * 100),
    }
    return log, metrics


def plot_part_c(log, metrics, save_path="results/part_c_ekf_performance.png"):
    """Generates 4-panel estimation consistency figure for Part C."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    t = log["t"]
    sight = t[log["marker_seen"] == 1][0] if np.any(log["marker_seen"] == 1) else None

    # Panel 1: Trajectory
    draw_arena(ax1)
    ax1.plot(log["x_true"], log["y_true"], "g-", lw=2, label="Truth")
    ax1.plot(log["x_est"], log["y_est"], "b--", lw=1.8, label="EKF")
    ax1.plot(log["x_odom"], log["y_odom"], "r:", lw=1.5, label="Odom")
    ax1.set_title("Trajectory: Truth vs EKF vs Odometry", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Pos error with 2-sigma bounds
    for err, sig, c, lbl in [(log["err_x"], log["sigma_x"], "blue", "x"), (log["err_y"], log["sigma_y"], "magenta", "y")]:
        ax2.plot(t, err * 100, color=c, lw=1.4, label=f"$e_{lbl}$")
        ax2.plot(t, 200 * sig, color=c, ls="--", lw=0.8, alpha=0.7)
        ax2.plot(t, -200 * sig, color=c, ls="--", lw=0.8, alpha=0.7)
        ax2.fill_between(t, -200 * sig, 200 * sig, color=c, alpha=0.08)
    if sight:
        ax2.axvline(sight, color="green", ls="-.", label=f"ArUco Lock ({sight:.1f}s)")
    ax2.set_ylabel("Error (cm)")
    ax2.set_title(r"Position Error & $2\sigma$ Bounds", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, ls="--", alpha=0.5)

    # Panel 3: Heading error with 2-sigma bounds
    ax3.plot(t, np.rad2deg(log["err_th"]), color="purple", lw=1.4, label=r"$e_\theta$")
    ax3.plot(t, 2 * np.rad2deg(log["sigma_th"]), "k--", lw=0.8, alpha=0.7)
    ax3.plot(t, -2 * np.rad2deg(log["sigma_th"]), "k--", lw=0.8, alpha=0.7)
    ax3.fill_between(t, -2 * np.rad2deg(log["sigma_th"]), 2 * np.rad2deg(log["sigma_th"]), color="purple", alpha=0.08)
    if sight:
        ax3.axvline(sight, color="green", ls="-.")
    ax3.set_ylabel("Error (deg)")
    ax3.set_title(r"Heading Error & $2\sigma$ Bounds", fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, ls="--", alpha=0.5)

    # Panel 4: Variance Trace contraction
    ax4.plot(t, log["sigma_x"]**2 + log["sigma_y"]**2, "k-", lw=1.5, label=r"$\text{Tr}(P_{\text{pos}})$")
    if sight:
        ax4.axvline(sight, color="green", ls="-.", label="Vision Lock")
    ax4.set_yscale("log")
    ax4.set_ylabel(r"Variance ($m^2$)")
    ax4.set_title("Covariance Contraction", fontweight="bold")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(True, ls="--", alpha=0.5)
    txt = f"Pos RMSE: {metrics['rmse_pos']*100:.1f}cm\nHead RMSE: {metrics['rmse_th']:.1f}°\nDock Err: {metrics['dock_err_dist']*100:.1f}cm\nIn-2σ: {metrics['in_2sigma_x']:.0f}%"
    ax4.text(0.05, 0.08, txt, transform=ax4.transAxes, fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for ax in (ax2, ax3, ax4):
        ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVED] {save_path}")


def plot_part_e(trials_data, save_path="results/part_e_robustness_trials.png"):
    """Generates 2-panel multi-trial robustness and convergence figure for Part E."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_arena(ax1)
    ax1.add_patch(plt.Rectangle((-1.8, -1.8), 3.6, 3.6, fill=False, ec="k", ls="--", lw=1.5, label="Walls"))
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(trials_data)))
    for i, (log, _) in enumerate(trials_data):
        ax1.plot(log["x_true"], log["y_true"], color=colors[i], lw=1.5)
        ax1.plot(log["x_true"][0], log["y_true"][0], "o", color=colors[i], ms=4)
    ax1.set_title(f"Part E: Multi-Trial Convergence ({len(trials_data)} Trials)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Summary bars
    x = np.arange(len(trials_data))
    rmses = [m["rmse_pos"] * 100 for _, m in trials_data]
    dock_errs = [m["dock_err_dist"] * 100 for _, m in trials_data]
    ax2.bar(x - 0.17, rmses, 0.35, label="Pos RMSE (cm)", color="royalblue")
    ax2.bar(x + 0.17, dock_errs, 0.35, label="Dock Err (cm)", color="forestgreen")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{i+1}" for i in x])
    ax2.set_ylabel("Error (cm)")
    ax2.set_title("Tracking & Docking Error per Trial", fontweight="bold")
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.5, axis="y")
    txt = f"Success: {sum(m['docked'] for _, m in trials_data)}/{len(trials_data)}\nMean RMSE: {np.mean(rmses):.1f}cm\nMean Dock Err: {np.mean(dock_errs):.1f}cm"
    ax2.text(0.04, 0.72, txt, transform=ax2.transAxes, fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVED] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated Evaluation Suite")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs("results/logs", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    robot, aruco, lidar, ekf = TurtleBotSim(), ArucoDetector(), LidarProcessor(), RobotEKF()
    trials_data, summary = [], []

    print(f"=== Running Evaluation ({args.trials} Trials) ===")
    for i in range(args.trials):
        x0 = np.array([rng.uniform(0.85, 1.25), rng.uniform(-0.50, 0.50), np.pi + rng.uniform(-np.deg2rad(35), np.deg2rad(35))])
        print(f"[T{i+1:02d}] Start: [{x0[0]:.2f}, {x0[1]:.2f}, {np.rad2deg(x0[2]):.0f}°] ... ", end="", flush=True)
        log, m = run_single_trial(robot, aruco, lidar, ekf, x0)
        trials_data.append((log, m))
        print(f"{'DOCKED' if m['docked'] else 'FAILED'} in {m['duration']:.1f}s | RMSE={m['rmse_pos']*100:.1f}cm | Err={m['dock_err_dist']*100:.1f}cm")

        # Save per-trial CSV
        save_cols = [c for c in COLS if c not in ("err_x", "err_y", "err_th")]
        data_to_save = np.column_stack([log[c] for c in save_cols])
        np.savetxt(f"results/logs/trial_{i+1:02d}.csv", data_to_save, delimiter=",", header=",".join(save_cols), comments="", fmt="%.4f")

        summary.append([i + 1, round(x0[0], 3), round(x0[1], 3), round(np.rad2deg(x0[2]), 1), m["docked"],
                        round(m["duration"], 2), round(m["rmse_pos"] * 100, 2), round(m["rmse_th"], 2),
                        round(m["dock_err_dist"] * 100, 2), round(m["dock_err_head"], 2),
                        round(m["in_2sigma_x"], 1), round(m["in_2sigma_y"], 1)])

    plot_part_c(trials_data[0][0], trials_data[0][1])
    plot_part_e(trials_data)

    sum_hdr = "trial_id,x0,y0,th0_deg,docked,duration_s,rmse_pos_cm,rmse_th_deg,dock_err_dist_cm,dock_err_head_deg,in_2sigma_x_pct,in_2sigma_y_pct"
    np.savetxt("results/summary_metrics.csv", summary, delimiter=",", header=sum_hdr, comments="", fmt="%s")
    print("[SAVED] results/summary_metrics.csv")


if __name__ == "__main__":
    main()
