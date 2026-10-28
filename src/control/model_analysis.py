import numpy as np

# Part A Modelling: State-Space, Controllability, and Observability Analysis

def analyze_controllability():
    """
    Evaluates the Lie Algebra Rank Condition (LARC) for non-holonomic unicycle kinematics:
      x_dot = v * cos(theta)
      y_dot = v * sin(theta)
      theta_dot = omega
    Vector fields:
      g1 = [cos(theta), sin(theta), 0]^T  (linear velocity input)
      g2 = [0, 0, 1]^T                   (angular velocity input)
    Lie bracket [g1, g2] = [sin(theta), -cos(theta), 0]^T
    """
    print("=======================================================")
    print(" PART A: CONTROLLABILITY ANALYSIS (LIE ALGEBRA / LARC)")
    print("=======================================================")
    
    # Test rank of Delta = [g1, g2, [g1, g2]] over all angles
    test_angles = np.linspace(-np.pi, np.pi, 36)
    ranks = []
    dets = []
    for th in test_angles:
        g1 = np.array([np.cos(th), np.sin(th), 0.0])
        g2 = np.array([0.0, 0.0, 1.0])
        bracket = np.array([np.sin(th), -np.cos(th), 0.0])
        Delta = np.column_stack([g1, g2, bracket])
        det_val = np.linalg.det(Delta)
        rank_val = np.linalg.matrix_rank(Delta)
        ranks.append(rank_val)
        dets.append(det_val)

    all_rank_3 = all(r == 3 for r in ranks)
    det_identity = np.allclose(dets, 1.0)
    print(f"Distribution matrix: Delta = [g1, g2, [g1, g2]]")
    print(f"Analytic determinant: det(Delta) = cos^2(theta) + sin^2(theta) = 1.0 != 0")
    print(f"Numerical rank check across 36 orientations: {set(ranks)}")
    print(f"Result: LARC satisfied (Rank = 3). System is small-time locally controllable: {all_rank_3}\n")
    return all_rank_3


def analyze_observability():
    """
    Evaluates observability rank with and without ArUco marker measurements:
      z = [r, phi]^T
    """
    print("=======================================================")
    print(" PART A: OBSERVABILITY ANALYSIS (EKF SENSING)")
    print("=======================================================")

    dt = 0.05
    v, th = 0.12, np.pi
    dock_pos = np.array([-1.52, 0.0])
    robot_pos = np.array([0.5, 0.1, th])

    # State transition Jacobian F
    F = np.eye(3)
    F[0, 2] = -v * np.sin(th) * dt
    F[1, 2] = v * np.cos(th) * dt

    # 1. With ArUco marker in FOV
    dx = dock_pos[0] - robot_pos[0]
    dy = dock_pos[1] - robot_pos[1]
    r = np.hypot(dx, dy)
    H_vis = np.array([
        [-dx / r, -dy / r, 0.0],
        [dy / (r**2), -dx / (r**2), -1.0]
    ])

    # Observability matrix O = [H; H*F; H*F^2]
    O_vis = np.vstack([H_vis, H_vis @ F, H_vis @ (F @ F)])
    rank_vis = np.linalg.matrix_rank(O_vis)

    # 2. When marker is occluded / out of camera FOV
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
