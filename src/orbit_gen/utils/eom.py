import numpy as np
from scipy.integrate import solve_ivp


def cr3bp_eom(t, w, mu_star):
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

    r1_mag = np.sqrt((x + m2) ** 2 + y**2 + z**2)
    r2_mag = np.sqrt((x - 1 + m2) ** 2 + y**2 + z**2)

    ax = 2 * vy + x - m1 * (x + m2) / r1_mag**3 - m2 * (x - 1 + m2) / r2_mag**3
    ay = -2 * vx + y - m1 * y / r1_mag**3 - m2 * y / r2_mag**3
    az = -m1 * z / r1_mag**3 - m2 * z / r2_mag**3

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
            -m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * z**2 / r1_mag**5
            + 3 * m2 * z**2 / r2_mag**5
        )

        Phi = np.reshape(w[6:], (6, 6))
        I = np.identity(3)
        A = np.array([[dxdx, dxdy, dxdz], [dxdy, dydy, dydz], [dxdz, dydz, dzdz]])
        Omega = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
        J = np.zeros([6, 6])
        J[:3, 3:] = I
        J[3:, :3] = A
        J[3:, 3:] = Omega
        dPhi = np.reshape(J @ Phi, (1, 36))[0]
        dw = np.append([vx, vy, vz, ax, ay, az], dPhi)
    else:
        dw = [vx, vy, vz, ax, ay, az]

    return dw


def propagate(free_var, mu_star, dense=True, r_min=None):
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
            CR3BP mass parameter.
        dense (bool, optional):
            If True, return 8 evenly spaced samples inside every integrator step
            (from the integrator's dense output); if False, only the step ends.
        r_min (float, optional):
            If given, stop the integration and raise RuntimeError when the
            trajectory comes closer than r_min (DU) to the secondary body. Near
            the body's centre the integrator would otherwise crawl.

    Returns:
        states (ndarray):
            Trajectory array. Shape is either (N, 6) for state-only propagation
            or (N, 42) when STM propagation is included.
        times (ndarray):
            Time vector of shape (N,) corresponding to integration output.

    Raises:
        RuntimeError:
            If the integration fails or comes within r_min of the secondary.
    """

    if len(free_var) > 4:
        x0 = np.append(
            [free_var[0], 0, free_var[1], 0, free_var[2], 0], free_var[4:]
        ).tolist()
    else:
        x0 = [free_var[0], 0, free_var[1], 0, free_var[2], 0]

    T = free_var[3]

    events = None
    if r_min is not None:

        def too_close(t, w, mu_star):
            return (w[0] - 1 + mu_star) ** 2 + w[1] ** 2 + w[2] ** 2 - r_min**2

        too_close.terminal = True
        events = too_close

    sol = solve_ivp(
        cr3bp_eom,
        [0, T],
        x0,
        args=(mu_star,),
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
        dense_output=dense,
        events=events,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    if sol.status == 1:
        raise RuntimeError("Trajectory came within r_min of the secondary body")

    if not dense:
        return sol.y.T, sol.t

    times = np.append(
        np.linspace(sol.t[:-1], sol.t[1:], 8, endpoint=False).T.ravel(), sol.t[-1]
    )
    return sol.sol(times).T, times


def constraint(free_var, mu_star, r_min=None):
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
        r_min (float, optional):
            Passed to `propagate`.

    Returns:
        Fx (ndarray, shape (3,)):
            Periodicity constraint residuals evaluated at T/2:
            typically enforcing symmetry conditions such as [y, vx, vz].
        state_half_T (ndarray, shape (6,)):
            State evaluated at T/2.
        Phi (ndarray, shape (6, 6)):
            State transition matrix evaluated at T/2.
    """

    states, _ = propagate(free_var, mu_star, dense=False, r_min=r_min)
    state = states[-1]

    Fx = np.array([state[1], state[3], state[5]])

    if len(state) > 7:
        Phi = np.reshape(state[6:], (6, 6))
    else:
        Phi = None

    state_half_T = state[0:6]

    return Fx, state_half_T, Phi


def jacobian(free_var, mu_star, state_half_T, Phi):
    """Compute Jacobian of periodicity constraints for differential correction

    Evaluates the sensitivity matrix dFx/dX for the CR3BP periodicity constraints
    used in the single-shooting differential corrector.

    Args:
        free_var (array_like):
            Free-variable vector [x0, z0, vy0, T/2].
        mu_star (float):
            CR3BP mass parameter (normalized secondary mass).
        state_half_T (ndarray, shape (6,)):
            State evaluated at T/2.
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

    dw = cr3bp_eom(free_var[-1], state_half_T, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])

    dFx = np.zeros([3, 4])
    dFx[:, 0:3] = phis
    dFx[:, 3] = ddT

    return dFx


def jacobi_constant(pos, vel, mu_star):
    """Compute the Jacobi constant in the circular restricted three-body problem

    Evaluates the Jacobi constant in the rotating frame for a given state. The
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

    r_secondary = np.array([1 - mu_star, 0, 0])
    r_primary = np.array([-mu_star, 0, 0])

    omega = (
        (pos[0] ** 2 + pos[1] ** 2) / 2
        + (1 - mu_star) / np.linalg.norm(pos - r_primary)
        + mu_star / np.linalg.norm(pos - r_secondary)
    )

    J = 2 * omega - np.dot(vel, vel)

    return J
