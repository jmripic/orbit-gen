import pytest
import astropy.units as u
from orbit_gen.utils import units


def test_time_round_trip(rng, cu_rand):
    # dimensional -> canonical -> dimensional
    dim_time = rng.uniform(0.1, 100.0) * u.day
    canonical_time = units.time_to_canonical(dim_time, cu_rand)
    dim_time_back = units.time_to_dimensional(canonical_time, cu_rand)
    assert dim_time_back.unit == u.day
    assert dim_time_back.value == pytest.approx(dim_time.value)


def test_time_round_trip_other_unit(rng, cu_rand):
    # Starting from a non-day unit should still round-trip correctly
    dim_time = rng.uniform(1000.0, 5000.0) * u.s
    canonical_time = units.time_to_canonical(dim_time, cu_rand)
    dim_time_back = units.time_to_dimensional(canonical_time, cu_rand)
    assert dim_time_back.to("s").value == pytest.approx(dim_time.to("s").value)


def test_pos_round_trip(rng, cu_rand):
    # dimensional -> canonical -> dimensional should return the original value
    dim_pos = rng.uniform(1.0e3, 1.0e6) * u.km
    canonical_pos = units.pos_to_canonical(dim_pos, cu_rand)
    dim_pos_back = units.pos_to_dimensional(canonical_pos, cu_rand)
    assert dim_pos_back.unit == u.m
    assert dim_pos_back.value == pytest.approx(dim_pos.to("m").value)


def test_pos_round_trip_other_unit(rng, cu_rand):
    # Starting from a non-meter unit should still round-trip correctly
    dim_pos = rng.uniform(1.0, 10.0) * u.AU
    canonical_pos = units.pos_to_canonical(dim_pos, cu_rand)
    dim_pos_back = units.pos_to_dimensional(canonical_pos, cu_rand)
    assert dim_pos_back.to("AU").value == pytest.approx(dim_pos.to("AU").value)


def test_vel_round_trip(rng, cu_rand):
    # dimensional -> canonical -> dimensional should return the original value
    dim_vel = rng.uniform(0.1, 10.0) * (u.km / u.s)
    canonical_vel = units.vel_to_canonical(dim_vel, cu_rand)
    dim_vel_back = units.vel_to_dimensional(canonical_vel, cu_rand)
    assert dim_vel_back.unit == (u.m / u.day)
    assert dim_vel_back.to("km/s").value == pytest.approx(dim_vel.value)


def test_vel_round_trip_other_unit(rng, cu_rand):
    # Starting from a non-standard velocity unit should still round-trip
    dim_vel = rng.uniform(1.0e3, 1.0e5) * (u.m / u.day)
    canonical_vel = units.vel_to_canonical(dim_vel, cu_rand)
    dim_vel_back = units.vel_to_dimensional(canonical_vel, cu_rand)
    assert dim_vel_back.to("m/d").value == pytest.approx(dim_vel.to("m/d").value)
