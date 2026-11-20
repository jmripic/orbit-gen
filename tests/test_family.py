from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from orbit_gen.family import Family
from orbit_gen.orbit import Orbit
from orbit_gen.utils import eom


def _make_family(
    mu_star, cu, r_primary=6378.1366 / 384400, r_secondary=0.004519771071800208
):
    fam = Family.__new__(Family)
    fam.mu_star = mu_star
    fam.m1 = 1 - mu_star
    fam.m2 = mu_star
    fam.r_primary = r_primary
    fam.r_secondary = r_secondary
    fam.orbits = []
    fam.cfg = SimpleNamespace(
        continuation=SimpleNamespace(max_iter=50, eps=1e-10, step=0.01),
        system=SimpleNamespace(canonical_units=cu, name="Earth-Moon"),
        ic=SimpleNamespace(name="halo"),
        verbose=True,
    )
    return fam


def test_correct_orbit_converges(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    X_perturbed = ic + np.array([1e-4, 1e-4, 1e-4, 1e-4])
    _, error, _ = fam._correct_orbit(X_perturbed, np.eye(6).flatten())
    assert error < fam.cfg.continuation.eps


def test_propagate_and_validate_no_intersection(mu_star, cu, ic):
    fam = _make_family(mu_star, cu, r_secondary=0.001)
    states, times, valid = fam._propagate_and_validate(ic.tolist())
    assert valid is True
    assert states.shape[1] == 6
    assert len(times) == len(states)


def test_propagate_and_validate_intersection(mu_star, cu, ic):
    fam = _make_family(mu_star, cu, r_secondary=0.5)
    _, _, valid = fam._propagate_and_validate(ic.tolist())
    assert valid is False


def test_continuation_step_returns_normalized_tangent(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    z = None
    _, z_new, success = fam._continuation_step(ic, z)
    assert success is True
    assert z_new.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(z_new), 1.0, atol=1e-10)


def test_correct_orbit_nonconvergence(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    fam.cfg.continuation.max_iter = 1
    X_perturbed = ic + np.array([1, 1, 1, 1])
    _, error, _ = fam._correct_orbit(X_perturbed, np.eye(6).flatten())
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
    assert "jacobi_constants" in data
    assert "stability_indices" in data
    np.testing.assert_allclose(data["mu_star"], mu_star)
    np.testing.assert_allclose(data["periods"][0], orbit.period)
    np.testing.assert_array_equal(data["jacobi_constants"], fam.jacobi_constants)
    np.testing.assert_array_equal(data["stability_indices"], fam.stability_indices)


def _run(fam, X0, max_solutions=20):
    fam.cfg.ic.as_list = list(X0)
    fam.cfg.continuation.max_solutions = max_solutions
    fam.cfg.continuation.max_period_days = float("inf")
    fam.cfg.verbose = False
    fam._run_continuation()


def test_continuation_step_first_tangent_increases_period(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    X, _, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    X_new, z_new, success = fam._continuation_step(X, None)
    assert success is True
    assert X_new[3] > X[3]
    np.testing.assert_allclose(np.linalg.norm(z_new), 1.0, atol=1e-10)


def test_continuation_step_converges(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    X, _, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    X_new, _, success = fam._continuation_step(X, None)
    assert success is True
    assert not np.allclose(X_new, X)
    _, error, _ = fam._correct_orbit(X_new, np.eye(6).flatten())
    assert error < fam.cfg.continuation.eps


def test_continuation_step_halves_step(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    X, _, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    X1, z1, _ = fam._continuation_step(X, None)
    fam.cfg.continuation.step = 0.3
    X2, _, success = fam._continuation_step(X1, z1)
    assert success is True
    step_used = z1 @ (np.array(X2) - np.array(X1))
    assert step_used < 0.3 - 1e-6
    halvings = float(np.log2(0.3 / step_used))
    assert halvings == pytest.approx(round(halvings))


def test_continuation_step_failure(mu_star, cu, ic, monkeypatch):
    fam = _make_family(mu_star, cu)
    X, _, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    z = np.array([0.0, 0.0, 0.0, 1.0])

    def fail(*args, **kwargs):
        raise RuntimeError("Trajectory came within r_min of the secondary body")

    monkeypatch.setattr(eom, "constraint", fail)
    with pytest.warns(UserWarning, match="did not converge"):
        X_new, z_new, success = fam._continuation_step(X, z)
    assert success is False
    np.testing.assert_array_equal(X_new, X)
    np.testing.assert_array_equal(z_new, z)


def test_correct_orbit_integration_failure(mu_star, cu, ic, monkeypatch):
    fam = _make_family(mu_star, cu)

    def fail(*args, **kwargs):
        raise RuntimeError("Trajectory came within r_min of the secondary body")

    monkeypatch.setattr(eom, "constraint", fail)
    _, error, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    assert error == np.inf


def test_correct_orbit_nan_residual(mu_star, cu, ic, monkeypatch):
    fam = _make_family(mu_star, cu)
    calls = []

    def nan_constraint(*args, **kwargs):
        calls.append(1)
        return np.full(3, np.nan), np.zeros(6), np.eye(6)

    monkeypatch.setattr(eom, "constraint", nan_constraint)
    _, error, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    assert len(calls) == 1
    assert error == np.inf


def test_discarded_orbits_count_toward_max_solutions(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    calls = []
    correct = fam._correct_orbit

    def counting(X, stm_identity):
        calls.append(1)
        return correct(X, stm_identity)

    fam._correct_orbit = counting
    fam._propagate_and_validate = lambda X: (np.zeros((2, 6)), np.zeros(2), False)
    with pytest.raises(RuntimeError, match="Continuation failed"):
        _run(fam, ic, max_solutions=3)
    assert len(calls) == 3


def test_equilibrium_stop(mu_star, cu):
    def ax(x):
        return x - (1 - mu_star) / (x + mu_star) ** 2 - mu_star / (x - 1 + mu_star) ** 2

    x_l2 = brentq(ax, 1.01, 1.3, xtol=1e-15)
    fam = _make_family(mu_star, cu)
    with pytest.warns(UserWarning, match="equilibrium"):
        with pytest.raises(RuntimeError, match="Continuation failed"):
            _run(fam, [x_l2, 0.0, 0.0, 1.0])


def test_bifurcation_guard(mu_star, cu):
    X0 = [1.1761468209649801, -0.065306608705016, -0.17649497452160506, 1.68991965467]
    fam = _make_family(mu_star, cu)
    with pytest.warns(UserWarning, match="bifurcation"):
        _run(fam, X0, max_solutions=40)
    z0 = np.array([orbit.z0 for orbit in fam.orbits])
    assert len(z0) > 0
    assert np.all(z0 / X0[1] > 1e-3)


def test_monodromy_symmetry_matches_full_period(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    _run(fam, ic, max_solutions=1)
    orbit = fam.orbits[0]
    M = orbit.monodromy
    assert M is not None
    full = Orbit(orbit.x0, orbit.z0, orbit.vy0, orbit.period, mu_star, cu)
    full.propagate(monodromy=True)
    assert full.monodromy is not None
    np.testing.assert_allclose(M, full.monodromy, atol=1e-8 * np.abs(M).max())
    assert abs(np.linalg.det(M) - 1) < 1e-8
    x0 = [orbit.x0, 0.0, orbit.z0, 0.0, orbit.vy0, 0.0]
    f = np.array(eom.cr3bp_eom(0.0, x0, mu_star))
    np.testing.assert_allclose(M @ f, f, atol=1e-8 * np.linalg.norm(f))


def test_monodromy_matches_jpl_stability_index(cu):
    mu_jpl = 1.215058560962404e-02
    X0 = [
        1.111343996268567,
        0.1933928533064504,
        -0.22201580631068876,
        2.789849359755705 / 2,
    ]
    fam = _make_family(mu_jpl, cu)
    _run(fam, X0, max_solutions=1)
    M = fam.orbits[0].monodromy
    assert M is not None
    lam_max = np.max(np.abs(np.linalg.eigvals(M)))
    np.testing.assert_allclose((lam_max + 1 / lam_max) / 2, 12.870946871458, rtol=1e-7)


def test_correction_passes_r_min(mu_star, cu, ic, monkeypatch):
    fam = _make_family(mu_star, cu)
    X, _, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    X1, z1, _ = fam._continuation_step(X, None)
    r_mins = []
    constraint = eom.constraint

    def recording(*args, **kwargs):
        r_mins.append(kwargs.get("r_min"))
        return constraint(*args, **kwargs)

    monkeypatch.setattr(eom, "constraint", recording)
    fam._correct_orbit(X1, np.eye(6).flatten())
    n_corrector = len(r_mins)
    fam._continuation_step(X1, z1)
    assert 0 < n_corrector < len(r_mins)
    assert all(r_min == 0.1 * fam.r_secondary for r_min in r_mins)


def test_bifurcation_guard_threshold(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    z0s = iter([0.1] + [5e-5] * 19)
    fam._correct_orbit = lambda X, stm_identity: (
        [X[0], next(z0s), X[2], X[3]],
        0.0,
        np.eye(6),
    )
    fam._propagate_and_validate = lambda X: (
        np.zeros((2, 6)),
        np.array([0.0, 2 * X[3]]),
        True,
    )
    fam._continuation_step = lambda X, z: (X, np.zeros(4), True)
    with pytest.warns(UserWarning, match="bifurcation"):
        _run(fam, ic)
    assert len(fam.orbits) == 1


def test_bifurcation_guard_threshold_not_reached(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    z0s = iter([0.1] + [2e-4] * 19)
    fam._correct_orbit = lambda X, stm_identity: (
        [X[0], next(z0s), X[2], X[3]],
        0.0,
        np.eye(6),
    )
    fam._propagate_and_validate = lambda X: (
        np.zeros((2, 6)),
        np.array([0.0, 2 * X[3]]),
        True,
    )
    fam._continuation_step = lambda X, z: (X, np.zeros(4), True)
    _run(fam, ic)
    assert len(fam.orbits) == 20


def test_propagate_and_validate_primary_intersection(mu_star, cu, ic):
    fam = _make_family(mu_star, cu, r_primary=1.0)
    _, _, valid = fam._propagate_and_validate(ic.tolist())
    assert valid is False


def test_continuation_step_same_family_rejection(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    X, _, _ = fam._correct_orbit(ic, np.eye(6).flatten())
    X1, z1, _ = fam._continuation_step(X, None)
    w = np.array([0.0, 1.0, 0.0, 0.0])
    w = w - (w @ z1) * z1
    z_bad = z1 + 0.5 * w / np.linalg.norm(w)
    z_bad = z_bad / np.linalg.norm(z_bad)
    with pytest.warns(UserWarning, match="same-family check"):
        X_new, _, success = fam._continuation_step(X1, z_bad)
    assert success is False
    np.testing.assert_array_equal(X_new, X1)


def test_period_guard(mu_star, cu, ic):
    fam = _make_family(mu_star, cu)
    with pytest.warns(UserWarning, match="Period below"):
        with pytest.raises(RuntimeError, match="Continuation failed"):
            _run(fam, [*ic[:3], 1e-12])


def test_correct_orbit_later_integration_failure(mu_star, cu, ic, monkeypatch):
    fam = _make_family(mu_star, cu)
    calls = []
    constraint = eom.constraint

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("Trajectory came within r_min of the secondary body")
        return constraint(*args, **kwargs)

    monkeypatch.setattr(eom, "constraint", fail_second)
    X_perturbed = ic + np.array([1e-3, 1e-3, 1e-3, 1e-3])
    _, error, _ = fam._correct_orbit(X_perturbed, np.eye(6).flatten())
    assert len(calls) == 2
    assert error == np.inf


def test_stability_index_matches_jpl(cu):
    mu_jpl = 1.215058560962404e-02
    X0 = [
        1.111343996268567,
        0.1933928533064504,
        -0.22201580631068876,
        2.789849359755705 / 2,
    ]

    fam = _make_family(mu_jpl, cu)
    _run(fam, X0, max_solutions=1)

    orbit = fam.orbits[0]

    np.testing.assert_allclose(
        orbit.stability_index,
        12.870946871458,
        rtol=1e-7,
    )


def test_jacobi_constant_matches_jpl(cu):
    mu_jpl = 1.215058560962404e-02
    X0 = [
        1.111343996268567,
        0.1933928533064504,
        -0.22201580631068876,
        2.789849359755705 / 2,
    ]

    fam = _make_family(mu_jpl, cu)
    _run(fam, X0, max_solutions=1)

    orbit = fam.orbits[0]

    np.testing.assert_allclose(
        orbit.jacobi_constant,
        3.0247424568048298e0,
        rtol=1e-6,
    )
