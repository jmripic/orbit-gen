import astropy.units as u
import numpy as np

from orbit_gen.setup.system_config import CanonicalUnits
from orbit_gen.utils import eom, units


class Orbit:
    """Single corrected periodic orbit in the CR3BP

    Represents a converged periodic orbit obtained from a differential correction
    procedure in the circular restricted three-body problem (CR3BP). The orbit
    is defined at a symmetry-plane crossing and can be numerically propagated to
    recover the full trajectory.

    Args:
        x0 (float):
            x-position at symmetry-plane crossing in canonical distance units (DU).
        z0 (float):
            z-position at symmetry-plane crossing in DU.
        vy0 (float):
            y-velocity at symmetry-plane crossing in canonical units (DU/TU).
        period (float):
            Full orbital period T in canonical time units (TU).
        mu_star (float):
            CR3BP mass ratio (normalized secondary mass parameter).

    Attributes:
        states (ndarray or None):
            Trajectory array of shape (N, 6) containing
            [x, y, z, vx, vy, vz] along the orbit. None until `propagate()` is called.
        times (ndarray or None):
            Time array of shape (N,) corresponding to `states`. None until
            `propagate()` is called.
        monodromy (ndarray or None):
            Monodromy matrix (state transition matrix over one full period) of
            shape (6, 6). None until `propagate(monodromy=True)` is called,
            unless set by `Family` during continuation.
    """

    def __init__(
        self,
        x0: float,
        z0: float,
        vy0: float,
        period: float,
        mu_star: float,
        canonical_units: CanonicalUnits,
    ):
        self.x0 = x0
        self.z0 = z0
        self.vy0 = vy0
        self.period = period
        self.mu_star = mu_star
        self.canonical_units = canonical_units

        self.states = None
        self.times = None
        self.monodromy = None

    @property
    def period_days(self) -> float:
        """Full orbital period in days."""
        return units.time_to_dimensional(self.period, self.canonical_units).to_value(
            u.day
        )

    @property
    def _free_var(self) -> list:
        """Full-period free variable vector [x0, z0, vy0, T]."""
        return [self.x0, self.z0, self.vy0, self.period]

    def propagate(self, monodromy: bool = False):
        """Integrate the trajectory for one full period.

        Stores results in self.states and self.times.

        Args:
            monodromy (bool, optional):
                If True, also integrate the state transition matrix and store
                Phi(T, 0) in self.monodromy.
        """
        if monodromy:
            stm_identity = np.reshape(np.eye(6), (1, 36))[0]
            states, self.times = eom.propagate(
                np.append(self._free_var, stm_identity), self.mu_star
            )
            self.states = states[:, 0:6]
            self.monodromy = np.reshape(states[-1, 6:], (6, 6))
        else:
            self.states, self.times = eom.propagate(self._free_var, self.mu_star)

    def plot(self, ax=None, **kwargs):
        """Plot the orbit in 3D

        Plots the trajectory of the orbit in a 3D matplotlib figure using the
        stored state history.

        Args:
            ax (mpl_toolkits.mplot3d.Axes3D, optional):
                Existing 3D axes object. If None, a new figure and 3D axes are
                created internally.
            **kwargs:
                Additional keyword arguments forwarded directly to `ax.plot()`,
                controlling styling (e.g., color, linewidth, alpha).

        Returns:
            ax (mpl_toolkits.mplot3d.Axes3D):
                The matplotlib 3D axes containing the plotted orbit.

        Raises:
            ImportError:
                If matplotlib is not installed (the `plotting` extra).
        """

        try:
            from matplotlib import pyplot as plt
        except ModuleNotFoundError as e:
            raise ImportError(
                'Plotting requires matplotlib: pip install "orbit-gen[plotting]"'
            ) from e

        if self.states is None:
            self.propagate()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        ax.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2], **kwargs)
        ax.set_xlabel("X [DU]")
        ax.set_ylabel("Y [DU]")
        ax.set_zlabel("Z [DU]")
        ax.set_aspect("equal", adjustable="datalim")

        return ax

    def save(self, path):
        """Save orbit initial conditions and period to an NPZ file

        Serializes the orbit’s defining parameters (initial state and period)
        to a NumPy `.npz` archive for later reuse or reconstruction.

        Args:
            path (str or pathlib.Path):
                Destination file path for the `.npz` archive. If the extension is
                omitted, `.npz` is assumed.
        """

        if self.states is None:
            self.propagate()

        np.savez(
            path,
            ic=self.states[0, :],
            period=self.period,
            mu_star=self.mu_star,
        )

    def __repr__(self):
        return (
            f"Orbit(x0={self.x0:.6f}, z0={self.z0:.6f}, "
            f"vy0={self.vy0:.6f}, T={self.period:.6f})"
        )
