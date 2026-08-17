import numpy as np
from types import SimpleNamespace

from orbit_gen.family import Family
from orbit_gen.orbit import Orbit


def _make_family(mu_star, cu, r_secondary=0.004519771071800208):
    fam = Family.__new__(Family)
    fam.mu_star = mu_star
    fam.m1 = 1 - mu_star
    fam.m2 = mu_star
    fam.r_secondary = r_secondary
    fam.orbits = []
    fam.cfg = SimpleNamespace(
        continuation=SimpleNamespace(max_iter=50, eps=1e-10, step=0.01),
        system=SimpleNamespace(canonical_units=cu, name="Earth-Moon"),
        ic=SimpleNamespace(name="halo"),
    )
    return fam


def test_correct_orbit_converges(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    X_perturbed = ic + np.array([1e-4, 1e-4, 1e-4, 1e-4])
    _, error = fam._correct_orbit(X_perturbed, np.eye(6).flatten())
    assert error < fam.cfg.continuation.eps


def test_propagate_and_validate_no_intersection(mu_star, cu, ic):
    fam = _make_family(mu_star, cu, r_secondary=0.001)
    states, times, valid = fam._propagate_and_validate(ic.tolist())
    assert valid is True
    assert states.shape[1] == 6
    assert len(times) == len(states)


def test_propagate_and_validate_intersection(mu_star, cu, ic):
    fam = _make_family(mu_star, cu, r_secondary=1.0)
    _, _, valid = fam._propagate_and_validate(ic.tolist())
    assert valid is False


def test_continuation_step_returns_normalized_tangent(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    z = np.array([0.0, 0.0, 0.0, -1.0])
    _, z_new, success = fam._continuation_step(ic, z)
    assert success is True
    assert z_new.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(z_new), 1.0, atol=1e-10)


def test_correct_orbit_nonconvergence(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    fam.cfg.continuation.max_iter = 1
    X_perturbed = ic + np.array([1, 1, 1, 1])
    _, error = fam._correct_orbit(X_perturbed, np.eye(6).flatten())
    assert error > fam.cfg.continuation.eps


def test_save_roundtrip(tmp_path, mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    fam.cfg.output_dir = tmp_path
    fam.cfg.output_path = tmp_path / "orbits.npz"

    orbit = Orbit(
        x0=ic[0],
        z0=ic[1],
        vy0=ic[2],
        period=2 * ic[3],
        mu_star=mu_star,
        canonical_units=cu,
    )
    orbit.states = np.zeros((10, 6))
    orbit.times = np.linspace(0, 2 * ic[3], 10)
    fam.orbits = [orbit]

    fam.save()

    data = np.load(fam.cfg.output_path)
    assert "states" in data
    assert "periods" in data
    assert "mu_star" in data
    np.testing.assert_allclose(data["mu_star"], mu_star)
    np.testing.assert_allclose(data["periods"][0], orbit.period)
