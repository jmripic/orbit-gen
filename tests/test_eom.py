import pytest
import numpy as np

from orbit_gen.utils import eom


def test_jacobi_constant_runs():
    # Ensures Jacobi constant returns a number
    pos = [1, 0.0, 0.0]
    vel = [0.0, 1, 0.0]
    result = eom.jacobi_constant(pos, vel, 1)
    assert isinstance(result, float)


def test_propagate_shapes_types(free_var, mu_star):
    # Ensures propagate returns the correct shapes and types
    states, times = eom.propagate(free_var, mu_star)

    assert isinstance(states, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert states.ndim == 2
    assert states.shape[1] == 6
    assert times.ndim == 1
    assert times.shape[0] == states.shape[0]


def test_jacobi_constant_conserved_along_trajectory(free_var, mu_star):
    # Ensures Jacobi constant remains roughly constant throughout
    # a trajectory.
    states, _ = eom.propagate(free_var, mu_star)

    jc_values = []
    for state in states:
        pos = state[0:3]
        vel = state[3:6]
        jc_values.append(eom.jacobi_constant(pos, vel, mu_star))

    jc_0 = jc_values[0]
    for value in jc_values:
        assert value == pytest.approx(jc_0)


def test_velocity_passthrough_and_autonomous(mu_star):
    # dx/dt, dy/dt, dz/dt must equal vx, vy, vz directly.
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    dw_t0 = eom.crtbp_eom(0.0, w, mu_star)
    dw_t100 = eom.crtbp_eom(100.0, w, mu_star)

    assert dw_t0[0] == w[3]
    assert dw_t0[1] == w[4]
    assert dw_t0[2] == w[5]
    np.testing.assert_array_equal(dw_t0, dw_t100)


def test_stm_jacobian(mu_star):
    # With Phi=I, crtbp_eom returns dPhi = J. Verify against numerical
    # finite difference across all 6 state components.
    w_state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    w = w_state + list(np.eye(6).flatten())

    J_analytic = np.reshape(eom.crtbp_eom(0.0, w, mu_star)[6:], (6, 6))

    eps = 1e-7
    f0 = np.array(eom.crtbp_eom(0.0, w_state, mu_star))
    J_numeric = np.zeros((6, 6))
    for i in range(6):
        w_pert = list(w_state)
        w_pert[i] += eps
        J_numeric[:, i] = (np.array(eom.crtbp_eom(0.0, w_pert, mu_star)) - f0) / eps

    np.testing.assert_allclose(J_analytic, J_numeric, atol=1e-5)


def test_constraint_output(free_var, mu_star):
    # Ensures constraint returns correct types/shapes
    # STM not included
    states, _ = eom.propagate(free_var, mu_star)
    Fx, state_half_T, Phi = eom.constraint(free_var, mu_star)

    assert isinstance(Fx, np.ndarray)
    assert Fx.shape == (3,)
    assert isinstance(state_half_T, np.ndarray)
    assert state_half_T.shape == (6,)
    assert Phi is None  # free_var has no STM appended -> state has len 6

    final_state = states[-1]
    np.testing.assert_array_equal(state_half_T, final_state[0:6])
    np.testing.assert_array_equal(
        Fx, np.array([final_state[1], final_state[3], final_state[5]])
    )

    # STM included
    free_var_stm = list(free_var) + list(np.eye(6).flatten())
    states, _ = eom.propagate(free_var_stm, mu_star)
    Fx, state_half_T, Phi = eom.constraint(free_var_stm, mu_star)

    final_state = states[-1]
    assert Phi is not None
    assert isinstance(Phi, np.ndarray)
    assert Phi.shape == (6, 6)
    np.testing.assert_array_equal(Phi, np.reshape(final_state[6:], (6, 6)))
    np.testing.assert_array_equal(state_half_T, final_state[0:6])


def test_constraint_time_derivative_finite_difference(free_var, mu_star):
    # Verify against numerical finite difference
    Fx0, state_half_T0, _ = eom.constraint(free_var, mu_star)

    eps = 1e-6
    free_var_pert = list(free_var)
    free_var_pert[3] += eps
    Fx_pert, _, _ = eom.constraint(free_var_pert, mu_star)

    dFx_dT_numeric = (Fx_pert - Fx0) / eps

    dFx_analytic = eom.jacobian(free_var, mu_star, state_half_T0, np.eye(6))
    dFx_dT_analytic = dFx_analytic[:, 3]

    np.testing.assert_allclose(dFx_dT_numeric, dFx_dT_analytic, atol=1e-5, rtol=1e-4)
