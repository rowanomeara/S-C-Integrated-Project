"""Hardware: Simulation vs Physical Hardware Benchmarking Plotter.

Generates quantitative comparison figures evaluating sim-to-real transfer:
1. Panel 1: Spatial trajectory overlay comparing simulation ground truth,
   simulation EKF estimate, and real TurtleBot3 physical path.
2. Panel 2: 2-sigma position uncertainty contraction over time, demonstrating
   consistent EKF convergence behavior across simulation and hardware.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Load simulation and physical flight logs to generate comparative figures."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sim_path = os.path.join(repo_root, "results", "logs", "trial_01.csv")
    real_path = os.path.join(repo_root, "results", "real_docking_log.csv")
    out_img = os.path.join(repo_root, "results", "part_e_sim_vs_real_comparison.png")

    if not os.path.exists(sim_path) or not os.path.exists(real_path):
        print("Missing log files for comparison (requires results/logs/trial_01.csv and results/real_docking_log.csv).")
        return

    sim_data = np.genfromtxt(sim_path, delimiter=",", names=True)
    real_data = np.genfromtxt(real_path, delimiter=",", names=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -------------------------------------------------------------
    # Panel 1: Trajectory Comparison in 2D World Plane
    # -------------------------------------------------------------
    dock_x, dock_y = -1.52, 0.0
    ax1.plot(sim_data["x_true"], sim_data["y_true"], "g--", label="Sim Ground Truth", alpha=0.8, linewidth=2)
    ax1.plot(sim_data["x_est"], sim_data["y_est"], "b-", label="Sim EKF Estimate", linewidth=2)
    ax1.plot(real_data["x_est"], real_data["y_est"], "r-o", label="Real TurtleBot3 Path", markersize=3, linewidth=2)

    # Docking station target and obstacle footprints
    ax1.plot(dock_x, dock_y, "ks", markersize=10, label="Dock Plate (-1.52, 0.0)")
    cyl = plt.Circle((-0.1, 0.0), 0.15, color="gray", alpha=0.5, label="Obstacle")
    ax1.add_patch(cyl)

    ax1.set_title("Trajectory Comparison: Simulation vs Physical TurtleBot3", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    ax1.set_aspect("equal")

    # -------------------------------------------------------------
    # Panel 2: EKF Uncertainty (2-Sigma Position Bounds) Contraction
    # -------------------------------------------------------------
    t_sim = sim_data["t"]
    t_real = real_data["time"]
    sig_sim = 2.0 * np.sqrt(sim_data["sigma_x"]**2 + sim_data["sigma_y"]**2)
    sig_real = 2.0 * np.sqrt(real_data["sigma_x"]**2 + real_data["sigma_y"]**2)

    ax2.plot(t_sim, sig_sim, "b-", label=r"Sim $2\sigma$ Uncertainty ($P_{xx}+P_{yy}$)", linewidth=2)
    ax2.plot(t_real, sig_real, "r-", label=r"Real Hardware $2\sigma$ Uncertainty", linewidth=2)

    ax2.set_title(r"EKF Covariance Contraction ($2\sigma$ Position Bound)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$2\sigma$ Position Uncertainty (m)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Generated comparison plot: {out_img}")


if __name__ == "__main__":
    main()
