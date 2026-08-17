import pytest
import logging
import numpy as np
import spiceypy as spice

from orbit_gen.orbit import Orbit
from orbit_gen.setup.system_config import CanonicalUnits
from orbit_gen.setup.kernels import ensure_kernels


@pytest.fixture(scope="session")
def mu_star():
    mu_star = 0.012150584395829193
    return mu_star


@pytest.fixture(scope="session")
def cu():
    cu = CanonicalUnits(dist_km=384400.0, time_days=4.34247988303745)
    return cu


@pytest.fixture(scope="session")
def ic():
    ic = np.array([0.857689, 0.182496, 0.256551, 2.434546173187467 / 2])
    return ic


@pytest.fixture(scope="session")
def orbit_propagate(mu_star, cu, ic):
    orbit = Orbit(
        x0=ic[0],
        z0=ic[1],
        vy0=ic[2],
        period=2 * ic[3],
        mu_star=mu_star,
        canonical_units=cu,
    )
    orbit.propagate()
    return orbit


@pytest.fixture(scope="session", autouse=True)
def rng():
    seed = np.random.SeedSequence().entropy
    logging.warning(f"RNG seed: {seed}")

    rng = np.random.default_rng(seed)

    return rng


@pytest.fixture(scope="module")
def free_var(rng):
    free_var = [
        rng.uniform(0.7, 0.9),
        rng.uniform(-0.1, 0.1),
        rng.uniform(0.1, 0.4),
        rng.uniform(1.0, 3.0),
    ]
    return free_var


@pytest.fixture(scope="module")
def cu_rand(rng):
    cu_rand = CanonicalUnits(
        dist_km=rng.uniform(1.0e5, 5.0e5),
        time_days=rng.uniform(10.0, 60.0),
    )
    return cu_rand


@pytest.fixture(scope="module")
def spice_furnish():
    kernel_paths = ensure_kernels().values()

    for path in kernel_paths:
        spice.furnsh(str(path))

    yield

    spice.kclear()
