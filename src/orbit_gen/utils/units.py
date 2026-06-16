import numpy as np
import astropy.units as u

_DU_KM = 3.844e5  # km per canonical distance unit
_TU_DAYS = 27.321582  # days per canonical time unit / (2*pi)
_DU_M = (_DU_KM * u.km).to("m").value


def time_to_canonical(dim_time):
    """Convert a dimensional time (astropy Quantity) to canonical TU."""
    return (dim_time.to("day").value / _TU_DAYS) * (2 * np.pi)


def time_to_dimensional(canonical_time):
    """Convert canonical time to days (astropy Quantity)."""
    return (canonical_time / (2 * np.pi)) * _TU_DAYS * u.day


def pos_to_canonical(dim_pos):
    """Convert a dimensional position (astropy Quantity) to canonical DU."""
    return dim_pos.to("m").value / _DU_M


def pos_to_dimensional(canonical_pos):
    """Convert canonical position to meters (astropy Quantity)."""
    return canonical_pos * _DU_M * u.m


def vel_to_canonical(dim_vel):
    """Convert a dimensional velocity (astropy Quantity) to canonical DU/TU."""
    dim_vel_mps = dim_vel.to("m/d").value
    return (dim_vel_mps / _DU_M) * (_TU_DAYS) / (2 * np.pi)


def vel_to_dimensional(canonical_vel):
    """Convert canonical velocity to m/day (astropy Quantity)."""
    return canonical_vel * (2 * np.pi) * _DU_M / _TU_DAYS * (u.m / u.day)
