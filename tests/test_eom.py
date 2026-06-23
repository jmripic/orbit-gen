from orbit_gen.utils import eom


def test_jacobi_constant_runs():
    # Ensures Jacobi constant returns a number
    pos = [1, 0.0, 0.0]
    vel = [0.0, 1, 0.0]
    result = eom.jacobi_constant(pos, vel, 1)
    assert isinstance(result, float)


def test_jacobi_constant_is_conserved_along_a_trajectory():
    # Ensures Jacobi constant remains roughly constant throughout
    # a trajectory.
    mu_star = 0.0121505856
    free_var = [0.8, 0.0, 0.2, 2.0]

    states, times = eom.propagate(free_var, mu_star)

    jc_values = []
    for state in states:
        pos = state[0:3]
        vel = state[3:6]
        jc_values.append(eom.jacobi_constant(pos, vel, mu_star))

    jc_0 = jc_values[0]
    for value in jc_values:
        assert abs(value - jc_0) < 1e-9
