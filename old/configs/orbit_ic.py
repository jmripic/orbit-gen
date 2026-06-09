from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrbitIC:
    """A single periodic orbit initial condition.

    Attributes:
        half_period (float):
            Half-period T/2 in canonical time units (TU)
        name (str):
            Label for the orbit family, including the system and orbit type
        vy0 (float):
            y-velocity in canonical velocity units (DU/TU)
        x0 (float):
            x-position in canonical distance units (DU)
        z0 (float):
            z-position in canonical distance units (DU)

    .. note::

        y0, vx0, and vz0 are always zero by the CR3BP symmetry condition. System
        is denoted by "EM" for Earth/Moon, and "SE" for Sun/Earth.

    """

    name: str
    x0: float
    z0: float
    vy0: float
    half_period: float

    @property
    def as_list(self) -> list:
        """Return free variables as [x0, z0, vy0, T/2] for the corrector."""
        return [self.x0, self.z0, self.vy0, self.half_period]

    @property
    def full_ic(self) -> list:
        """Return the full 7-element IC [x0,0,z0,0,vy0,0,T/2] for reference."""
        return [self.x0, 0.0, self.z0, 0.0, self.vy0, 0.0, self.half_period]
