from dataclasses import dataclass


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


EM_SYSTEM = SystemConfig(
    name="EM",
    primary="Earth",
    secondary="Moon",
)

SE_SYSTEM = SystemConfig(
    name="SE",
    primary="Sun",
    secondary="Earth",
)
