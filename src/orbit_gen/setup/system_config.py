from dataclasses import dataclass
import astropy.units as u


@dataclass(frozen=True)
class CanonicalUnits:
    """Characteristic length/time scales for a specific CR3BP system."""

    dist_km: float  # canonical distance [km]
    time_days: float | None = None  # canonical time / (2*pi) [days]

    @property
    def du_m(self) -> float:
        return (self.dist_km * u.km).to("m").value


@dataclass
class SystemConfig:
    """Determines parameters based on the CR3BP system

    Attributes:
        name (str):
            Label for the system
        primary (str):
            Primary body in the CR3BP
        secondary (str):
            Secondary body in the CR3BP

    .. note::

        "SE" for Sun/Earth "EM" for Earth/Moon, "Sun" for sun, "Earth" for Earth
        "Moon" for moon

    """

    name: str
    primary: str
    secondary: str
    canonical_units: CanonicalUnits


EM_SYSTEM = SystemConfig(
    name="EM",
    primary="Earth",
    secondary="Moon",
    canonical_units=CanonicalUnits(dist_km=3.844e5),
)

SE_SYSTEM = SystemConfig(
    name="SE",
    primary="Sun",
    secondary="Earth",
    canonical_units=CanonicalUnits(dist_km=u.au.to("km")),
)
