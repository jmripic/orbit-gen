import numpy as np

from orbit_gen.orbit import Orbit


def _make_orbit(mu_star, ic, cu):
    return Orbit(
        x0=ic[0],
        z0=ic[1],
        vy0=ic[2],
        period=2 * ic[3],
        mu_star=mu_star,
        canonical_units=cu,
    )


def test_states_none_before_propagate(mu_star, ic, cu):
    orbit = _make_orbit(mu_star, ic, cu)
    assert orbit.states is None
    assert orbit.times is None


def test_propagate_populates_states(mu_star, ic, cu):
    orbit = _make_orbit(mu_star, ic, cu)
    orbit.propagate()
    assert orbit.states is not None
    assert orbit.times is not None
    assert orbit.states.shape[1] == 6
    assert len(orbit.times) == len(orbit.states)


def test_period_days(mu_star, ic, cu):
    orbit = _make_orbit(mu_star, ic, cu)
    assert orbit.period_days > 0
    np.testing.assert_allclose(orbit.period_days, 2 * ic[3] * cu.time_days, rtol=1e-12)


def test_free_var_matches_inputs(mu_star, ic, cu):
    orbit = _make_orbit(mu_star, ic, cu)
    assert orbit._free_var == [ic[0], ic[1], ic[2], 2 * ic[3]]


def test_save_triggers_propagate_if_needed(tmp_path, mu_star, ic, cu):
    # save() should call propagate() automatically if states is None.
    orbit = _make_orbit(mu_star, ic, cu)
    assert orbit.states is None
    orbit.save(tmp_path / "orbit.npz")
    assert orbit.states is not None


def test_save_roundtrip(tmp_path, mu_star, ic, cu):
    orbit = _make_orbit(mu_star, ic, cu)
    path = tmp_path / "orbit.npz"
    orbit.save(path)

    data = np.load(path)
    assert "ic" in data
    assert "period" in data
    assert "mu_star" in data
    np.testing.assert_allclose(data["period"], 2 * ic[3])
    np.testing.assert_allclose(data["mu_star"], mu_star)
    np.testing.assert_array_equal(data["ic"], orbit.states[0, :])
