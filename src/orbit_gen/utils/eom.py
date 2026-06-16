import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def crtbp_eom(t, w, mu_star):
    """Equations of motion for the CR3BP in the rotating frame with optional STM propagation

    Computes the time derivative of the state in the circular restricted three-body problem
    (CR3BP). Supports both pure state propagation and augmented state propagation including
    a flattened 6×6 state transition matrix (STM).

    Args:
        t (float):
            Time variable (required by ODE integrators such as `scipy.integrate.solve_ivp`
            but not explicitly used in the CR3BP formulation).
        w (array_like):
            State vector. Two supported formats:
                - len == 6: [x, y, z, vx, vy, vz]
                - len == 42: augmented state including flattened 6×6 STM.
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).

    Returns:
        dw (ndarray):
            Time derivative of the input state vector (and STM if present),
            matching the shape of `w`.
    """

    [x, y, z, vx, vy, vz] = w[:6]

    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O = r1 * np.array([-1, 0, 0])
    r_2O = r2 * np.array([1, 0, 0])

    r_PO = np.array([x, y, z])
    v_PO = np.array([vx, vy, vz])

    r_P1 = r_PO - r_1O
    r_P2 = r_PO - r_2O
    r1_mag = np.linalg.norm(r_P1)
    r2_mag = np.linalg.norm(r_P2)

    F_g = -m1 / r1_mag**3 * r_P1 - m2 / r2_mag**3 * r_P2

    e3 = np.array([0, 0, 1])
    a_PO = F_g - 2 * np.cross(e3, v_PO) - np.cross(e3, np.cross(e3, r_PO))
    ax, ay, az = a_PO

    if len(w) > 7:
        dxdx = (
            1
            - m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * (x + m2) ** 2 / r1_mag**5
            + 3 * m2 * (x - 1 + m2) ** 2 / r2_mag**5
        )
        dxdy = 3 * m1 * (x + m2) * y / r1_mag**5 + 3 * m2 * (x - 1 + m2) * y / r2_mag**5
        dxdz = 3 * m1 * (x + m2) * z / r1_mag**5 + 3 * m2 * (x - 1 + m2) * z / r2_mag**5
        dydy = (
            1
            - m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * y**2 / r1_mag**5
            + 3 * m2 * y**2 / r2_mag**5
        )
        dydz = 3 * m1 * y * z / r1_mag**5 + 3 * m2 * y * z / r2_mag**5
        dzdz = (
            1
            - m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * z**2 / r1_mag**5
            + 3 * m2 * z**2 / r2_mag**5
        )

        Phi = np.reshape(w[6:], (6, 6))
        Z = np.zeros([3, 3])
        I = np.identity(3)
        A = np.array([[dxdx, dxdy, dxdz], [dxdy, dydy, dydz], [dxdz, dydz, dzdz]])
        J = np.block([[Z, I], [A, Z]])
        dPhi = np.reshape(J @ Phi, (1, 36))[0]
        dw = np.append([vx, vy, vz, ax, ay, az], dPhi)
    else:
        dw = [vx, vy, vz, ax, ay, az]

    return dw


def propagate(free_var, mu_star):
    """Integrate the CR3BP equations of motion over one full orbital period

    Performs numerical propagation of the CR3BP dynamics using the provided
    initial free variables. Optionally propagates the state transition matrix
    (STM) if included in the input.

    Args:
        free_var (array_like):
            Initial condition vector. Supported formats:
                - [x0, z0, vy0, T]
                - [x0, z0, vy0, T, STM_flat] where STM_flat has length 36
                (flattened 6×6 STM, total state dimension = 42).
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).

    Returns:
        states (ndarray):
            Trajectory array. Shape is either (N, 6) for state-only propagation
            or (N, 42) when STM propagation is included.
        times (ndarray):
            Time vector of shape (N,) corresponding to integration output.
    """

    if len(free_var) > 4:
        x0 = np.append(
            [free_var[0], 0, free_var[1], 0, free_var[2], 0], free_var[4:]
        ).tolist()
    else:
        x0 = [free_var[0], 0, free_var[1], 0, free_var[2], 0]

    T = free_var[3]

    sol = solve_ivp(crtbp_eom, [0, T], x0, args=(mu_star,), rtol=1e-12, atol=1e-12)
    return sol.y.T, sol.t


def constraint(free_var, mu_star):
    """Evaluate periodicity constraints for CR3BP symmetric periodic orbit

    Computes the constraint vector enforcing periodicity at the half-period
    crossing and returns the associated state transition matrix (STM) evaluated
    at T/2.

    Args:
        free_var (array_like):
            Free-variable vector of the form [x0, z0, vy0, T/2] optionally
            followed by a flattened 6×6 state transition matrix (STM).
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).

    Returns:
        Fx (ndarray, shape (3,)):
            Periodicity constraint residuals evaluated at T/2:
            typically enforcing symmetry conditions such as [y, vx, vz].
        Phi (ndarray, shape (6, 6)):
            State transition matrix evaluated at T/2.
    """

    states, _ = propagate(free_var, mu_star)
    state = states[-1]

    Fx = np.array([state[1], state[3], state[5]])

    if len(state) > 7:
        Phi = np.reshape(state[6:], (6, 6))
    else:
        Phi = None

    return Fx, Phi


def jacobian(free_var, mu_star, m1, m2, Phi):
    """Compute Jacobian of periodicity constraints for differential correction

    Evaluates the sensitivity matrix dFx/dX for the CR3BP periodicity constraints
    used in the single-shooting differential corrector.

    Args:
        free_var (array_like):
            Free-variable vector [x0, z0, vy0, T/2].
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).
        m1 (float):
            Normalized primary mass (1 - mu_star).
        m2 (float):
            Normalized secondary mass (mu_star).
        Phi (ndarray, shape (6, 6)):
            State transition matrix evaluated at T/2.

    Returns:
        dFx (ndarray, shape (3, 4)):
            Jacobian matrix of constraint residuals with respect to free variables.
    """

    phis = np.array(
        [
            [Phi[1, 0], Phi[1, 2], Phi[1, 4]],
            [Phi[3, 0], Phi[3, 2], Phi[3, 4]],
            [Phi[5, 0], Phi[5, 2], Phi[5, 4]],
        ]
    )

    X = [free_var[0], 0, free_var[1], 0, free_var[2], 0]
    dw = crtbp_eom(free_var[-1], X, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])

    dFx = np.zeros([3, 4])
    dFx[:, 0:3] = phis
    dFx[:, 3] = ddT

    return dFx


def continuation_constraint(w, z, sol_predictor, mu_star):
    """System of equations for pseudo-arclength continuation (for root finding)

    Defines the augmented nonlinear system used in pseudo-arclength continuation,
    typically solved using `fsolve`. Combines the CR3BP periodicity constraints
    with the arclength constraint relative to a predictor step.

    Args:
        w (array_like):
            Current free-variable vector [x0, z0, vy0, T/2].
        z (ndarray):
            Tangent direction vector at the current solution point.
        sol_predictor (ndarray):
            Predicted solution from the continuation predictor step.
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).

    Returns:
        ndarray, shape (4,):
            Residual vector combining periodicity constraints and arclength
            condition, suitable for nonlinear root solving.
    """

    Fx, _ = constraint(w, mu_star)
    zeq = z.T @ (w - sol_predictor)
    return np.append(Fx, zeq)


def jacobi_constant(pos, vel, mu_star):
    """Compute the Jacobi constant in the circular restricted three-body problem

    Evaluates the Jacobi integral in the rotating frame for a given state. The
    Jacobi constant is conserved along CR3BP trajectories and is commonly used
    as an invariant for validation and classification of orbits.

    Args:
        pos (array_like, shape (3,)):
            Position vector [x, y, z] in canonical rotating-frame coordinates.
        vel (array_like, shape (3,)):
            Velocity vector [vx, vy, vz] in canonical units.
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).

    Returns:
        float:
            Value of the Jacobi constant for the given state.
    """

    r_Mbary = np.array([1 - mu_star, 0, 0])
    r_Ebary = np.array([-mu_star, 0, 0])

    KE = np.dot(vel, vel) / 2
    U1 = -(pos[0] ** 2 + pos[1] ** 2) / 2
    U2 = -(
        (1 - mu_star) / np.linalg.norm(pos - r_Ebary)
        + mu_star / np.linalg.norm(pos - r_Mbary)
    )
    return KE + U1 + U2
