import spiceypy as spice
import pytest


def test_spice_sun_gm(spice_furnish):
    gm_sun = spice.bodvrd("Sun", "GM", 1)[1][0]

    assert gm_sun == pytest.approx(1.327e11, abs=1e9)


def test_spice_earth_gm(spice_furnish):
    gm_earth = spice.bodvrd("Earth", "GM", 1)[1][0]

    assert gm_earth == pytest.approx(398600, abs=1e3)


def test_spice_moon_gm(spice_furnish):
    gm_moon = spice.bodvrd("Moon", "GM", 1)[1][0]

    assert gm_moon == pytest.approx(4902.8, abs=1)


def test_spice_earth_radius(spice_furnish):
    radius_earth = spice.bodvrd("Earth", "RADII", 3)[1][0]

    assert radius_earth == pytest.approx(6371, abs=10)


def test_spice_moon_radius(spice_furnish):
    radius_moon = spice.bodvrd("Moon", "RADII", 3)[1][0]

    assert radius_moon == pytest.approx(1737, abs=10)
