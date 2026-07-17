from ..setup.system_config import CanonicalUnits
import astropy.units as u


def time_to_canonical(dim_time, cu: CanonicalUnits):
    return dim_time.to("day").value / cu.time_days


def time_to_dimensional(canonical_time, cu: CanonicalUnits):
    return canonical_time * cu.time_days * u.day


def pos_to_canonical(dim_pos, cu: CanonicalUnits):
    return dim_pos.to("m").value / cu.du_m


def pos_to_dimensional(canonical_pos, cu: CanonicalUnits):
    return canonical_pos * cu.du_m * u.m


def vel_to_canonical(dim_vel, cu: CanonicalUnits):
    dim_vel_mps = dim_vel.to("m/d").value
    return (dim_vel_mps / cu.du_m) * cu.time_days


def vel_to_dimensional(canonical_vel, cu: CanonicalUnits):
    return canonical_vel * cu.du_m / cu.time_days * (u.m / u.day)
