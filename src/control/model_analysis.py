"""Model Analysis: Controllability and Observability.

This module validates non-holonomic unicycle kinematics under UTS 41014 Part A:
1. Controllability via the Lie Algebra Rank Condition (LARC).
2. Observability via the rank of the observability matrix with and without visual landmarks.
"""

import numpy as np


def analyze_controllability():
    """Evaluate the Lie Algebra Rank Condition (LARC) for unicycle kinematics.

    Kinematics:
      x_dot = v * cos(theta)
      y_dot = v * sin(theta)
      theta_dot = omega

    Linearisation around zero velocities yields an uncontrollable system (Kalman rank < 3).
    We evaluate non-linear controllability through vector fields g1 and g2:
      g1 = [cos(theta), sin(theta), 0]^T   (forward drive input)
      g2 = [0, 0, 1]^T                    (yaw rotation input)

    The Lie bracket [g1, g2] represents sideways motion (parallel parking):
      [g1, g2] = (d g2 / d x) * g1 - (d g1 / d x) * g2
               = [sin(theta), -cos(theta), 0]^T

    The distribution matrix Delta = [g1, g2, [g1, g2]] has determinant:
      det(Delta) = cos^2(theta) + sin^2(theta) = 1.0 != 0
    Because det(Delta) = 1 everywhere, the distribution spans R^3 for all orientations.
    """
    print("=======================================================")
    print(" PART A: CONTROLLABILITY ANALYSIS (LIE ALGEBRA / LARC)")
    print("=======================================================")

    # Test the rank and determinant of Delta across 36 orientations spanning [-pi, pi]
    test_angles = np.linspace(-np.pi, np.pi, 36)
    ranks = []
    dets = []
    for th in test_angles:
        # Control vector field 1: forward translation along current heading
        g1 = np.array([np.cos(th), np.sin(th), 0.0])
        # Control vector field 2: pure in-place yaw rotation
        g2 = np.array([0.0, 0.0, 1.0])
        # Lie bracket [g1, g2]: generates lateral acceleration without direct sideways actuators
        bracket = np.array([np.sin(th), -np.cos(th), 0.0])

        # Form distribution matrix Delta = [g1, g2, [g1, g2]]
        Delta = np.column_stack([g1, g2, bracket])
        det_val = np.linalg.det(Delta)
        rank_val = np.linalg.matrix_rank(Delta)
        ranks.append(rank_val)
        dets.append(det_val)

    all_rank_3 = all(r == 3 for r in ranks)
    print(f"Distribution matrix: Delta = [g1, g2, [g1, g2]]")
    print(f"Analytic determinant: det(Delta) = cos^2(theta) + sin^2(theta) = 1.0 != 0")
    print(f"Numerical rank check across 36 orientations: {set(ranks)}")
    print(f"Result: LARC satisfied (Rank = 3). System is small-time locally controllable: {all_rank_3}\n")
    return all_rank_3


def analyze_observability():
    """Evaluate discrete-time observability rank with and without ArUco landmark sightings.

    State vector: x = [x, y, theta]^T
    State transition Jacobian: F = df/dx
    Measurement vector: z = [r, phi]^T (range and bearing to fixed docking landmark)
    Measurement Jacobian: H = dh/dx

    Observability matrix for a 3-state system:
      O = [ H; H*F; H*F^2 ]
    If rank(O) == 3, the state vector is fully observable.
    If rank(O) == 0 (e.g., marker occluded), dead reckoning drifts without bound.
    """
    print("=======================================================")
    print(" PART A: OBSERVABILITY ANALYSIS (EKF SENSING)")
    print("=======================================================")

    dt = 0.05
    v, th = 0.12, np.pi
    dock_pos = np.array([-1.52, 0.0])
    robot_pos = np.array([0.5, 0.1, th])

    # State transition Jacobian F for constant velocity motion
    F = np.eye(3)
    F[0, 2] = -v * np.sin(th) * dt
    F[1, 2] = v * np.cos(th) * dt

    # Case 1: ArUco marker visible in camera field of view
    # Distance and line-of-sight vector from robot pose to landmark
    dx = dock_pos[0] - robot_pos[0]
    dy = dock_pos[1] - robot_pos[1]
    r = np.hypot(dx, dy)

    # Measurement Jacobian H_vis = dh/dx for polar range and bearing:
    # dh1/dx = [-dx/r, -dy/r, 0]
    # dh2/dx = [dy/r^2, -dx/r^2, -1]
    H_vis = np.array([
        [-dx / r, -dy / r, 0.0],
        [dy / (r**2), -dx / (r**2), -1.0]
    ])

    # Observability matrix over 3 discrete time steps
    O_vis = np.vstack([H_vis, H_vis @ F, H_vis @ (F @ F)])
    rank_vis = np.linalg.matrix_rank(O_vis)

    # Case 2: Marker occluded or outside camera field of view (H = 0)
    H_blind = np.zeros((2, 3))
    O_blind = np.vstack([H_blind, H_blind @ F, H_blind @ (F @ F)])
    rank_blind = np.linalg.matrix_rank(O_blind)

    print(f"Case 1 (ArUco in FOV):     Observability Matrix Rank = {rank_vis} / 3 -> Full State Observable")
    print(f"Case 2 (ArUco Occluded):   Observability Matrix Rank = {rank_blind} / 3 -> Unobservable Drift")
    print("Conclusion: Global pose [x, y, theta] is fully observable only when the visual landmark is tracked.\n")
    return rank_vis == 3 and rank_blind == 0


def main():
    controllable = analyze_controllability()
    observable = analyze_observability()
    print(f"All Part A verification checks passed: {controllable and observable}")


if __name__ == "__main__":
    main()
