import numpy as np
import spiceypy as spice
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import astropy.units as u

from orbit_gen.utils import eom, units
from orbit_gen.orbit import Orbit
from orbit_gen.configs.config import RunConfig


class Family:
    """Family of CR3BP periodic orbits

    Generates a family of periodic orbits in the circular restricted
    three-body problem (CR3BP).

    Args:
        cfg (RunConfig):
            Full run configuration.

    Attributes:
        mu_star (float):
            CR3BP mass ratio.
        m1 (float):
            Normalized primary mass (1 - mu_star).
        m2 (float):
            Normalized secondary mass (mu_star).
        r_secondary (float):
            Radius of the secondary body in canonical distance units (DU).
        orbits (list[Orbit]):
            Collection of valid periodic orbit solutions obtained.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.orbits: list[Orbit] = []

        self._setup_constants()
        self._run_continuation()

    def _setup_constants(self):
        """Load SPICE kernel and compute CR3BP constants from system config."""
        for path in self.cfg.kernel_paths.values():
            spice.furnsh(str(path))

        sys = self.cfg.system
        gm_primary = spice.bodvrd(sys.primary, "GM", 1)[1][0]
        gm_secondary = spice.bodvrd(sys.secondary, "GM", 1)[1][0]

        self.mu_star = gm_secondary / (gm_primary + gm_secondary)
        self.m1 = 1 - self.mu_star
        self.m2 = self.mu_star

        radii = spice.bodvrd(sys.secondary, "RADII", 3)[1][0]
        self.r_secondary = units.pos_to_canonical(radii * u.km)

    def _run_continuation(self):
        """Outer pseudo-arclength loop. Populates self.orbits."""
        cont = self.cfg.continuation
        Tp_lim = units.time_to_canonical(cont.max_period_days * u.d)
        stm_identity = np.reshape(np.eye(6), (1, 36))[0]

        # Free variables: [x0, z0, vy0, T/2]
        X = self.cfg.ic.as_list
        z = np.array([0.0, 0.0, 0.0, -1.0])

        while X[-1] * 2 < Tp_lim and len(self.orbits) < cont.max_solutions:
            X, error = self._correct_orbit(X, stm_identity)

            if X[-1] < 0:
                print("Negative period.")
                break
            if error > cont.eps:
                print("Corrector did not converge.")
                break

            period_days = units.time_to_dimensional(2 * X[-1]).to_value(u.day)
            print(f"Period: {period_days:.4f} days")

            states, times, valid = self._propagate_and_validate(X)

            if valid:
                orbit = Orbit(
                    x0=X[0],
                    z0=X[1],
                    vy0=X[2],
                    period=times[-1],
                    mu_star=self.mu_star,
                )
                # Store already-computed trajectory so propagate() isn't needed again
                orbit.states = states
                orbit.times = times
                self.orbits.append(orbit)

            X, z, success = self._continuation_step(X, z)
            if not success:
                break

            print(f"Orbits found: {len(self.orbits)}\n")

        if not self.orbits:
            raise RuntimeError("All discovered orbits intersect the secondary body")

    def _correct_orbit(self, X: list, stm_identity: np.ndarray) -> tuple[list, float]:
        """Single-shooting differential corrector

        Newton-style differential correction using a single-shooting formulation
        for periodic orbit boundary condition enforcement.

        Args:
            X (array-like):
                Free-variable vector of the form [x0, z0, vy0, T/2].
            stm_identity (ndarray):
                Flattened 6x6 identity matrix used to initialize/propagate the
                state transition matrix.

        Returns:
            X (ndarray):
                Corrected free-variable vector after convergence.
            error (float):
                Final residual norm ||F(X)|| at termination.
        """

        cont = self.cfg.continuation
        error = np.inf

        for i in range(cont.max_iter):
            X_full = np.append(X, stm_identity).tolist()

            Fx, Phi = eom.constraint(X_full, self.mu_star)
            error = np.linalg.norm(Fx)

            print(f"  iter {i+1:2d} | error = {error:.3e}")

            if error < cont.eps:
                break

            dFx = eom.jacobian(X, self.mu_star, self.m1, self.m2, Phi)
            X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

        return X, error

    def _propagate_and_validate(self, X: list) -> tuple[np.ndarray, np.ndarray, bool]:
        """Propagate one full orbital period and check secondary-body intersection

        Numerically integrates the trajectory over a full orbital period from the
        given initial conditions, then evaluates whether the trajectory intersects
        or enters the secondary body's exclusion region.

        Args:
            X (array-like):
                Free-variable vector [x0, z0, vy0, T/2] defining the initial state
                and half-period used for symmetric propagation.

        Returns:
            states (ndarray):
                Propagated state history with shape (N, 6), where each row is
                [x, y, z, vx, vy, vz].
            times (ndarray):
                Time samples corresponding to each state, shape (N,).
            valid (bool):
                False if the trajectory intersects the secondary body or violates
                the exclusion radius; True otherwise.
        """

        X_full = X.copy()
        X_full[-1] = 2 * X_full[-1]

        states, times = eom.propagate(X_full, self.mu_star)
        positions = states[:, 0:3]
        r_mag = np.linalg.norm(positions, axis=1)

        perilune_km = units.pos_to_dimensional(min(r_mag)).to_value(u.km)
        print(f"  Perilune: {perilune_km:.2f} km")

        valid = not np.any(r_mag < self.r_secondary)
        if not valid:
            print("  Intersects secondary — discarding.")

        return states, times, valid

    def _continuation_step(
        self, X: list, z: np.ndarray
    ) -> tuple[list, np.ndarray, bool]:
        """Advance one pseudo-arclength continuation step along an orbit family

        Performs a predictor-corrector step along the solution manifold using a
        pseudo-arclength continuation method. Advances the current solution estimate
        and updates the tangent direction along the family.

        Args:
            X (ndarray):
                Current free-variable vector at the current solution point.
            z (ndarray):
                Current tangent direction vector along the continuation manifold.

        Returns:
            X_new (ndarray):
                Predicted/corrected free-variable vector for the next orbit.
            z_new (ndarray):
                Updated tangent direction at the new solution point.
            success (bool):
                False if the Jacobian is singular or the correction step fails;
                True otherwise.
        """

        sol_predictor = X + z * self.cfg.continuation.step

        fss = fsolve(
            eom.continuation_constraint,
            X,
            args=(z, sol_predictor, self.mu_star),
            full_output=True,
            xtol=1e-6,
        )
        X_new = fss[0]

        # Reconstruct Jacobian
        Q = fss[1]["fjac"]
        Rs = fss[1]["r"]
        R = np.zeros((4, 4))
        idx, col = np.triu_indices(4, k=0)
        R[idx, col] = Rs
        J = Q.T @ R

        try:
            z_new = np.linalg.inv(J) @ z
            z_new = z_new / np.linalg.norm(z_new)
            return X_new, z_new, True
        except np.linalg.LinAlgError:
            print("Singular Jacobian — stopping continuation.")
            return X_new, z, False

    def plot(self, show: bool = True):
        """Plot all orbits in the family on a single 3D figure

        Generates a 3D visualization of all stored periodic orbits in the family,
        typically in canonical CR3BP coordinates. Each orbit is plotted in the same
        figure for qualitative comparison of geometry across the continuation family.

        Args:
            show (bool):
                If True, displays the figure via matplotlib.pyplot.show().
                If False, the figure is created but not rendered interactively.
        """

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for orbit in self.orbits:
            orbit.plot(ax=ax)

        ax.set_title(f"{self.cfg.ic.name} family — {len(self.orbits)} orbits")

        if show:
            plt.show()

        return ax

    def save(self):
        """Save all orbit ICs, periods, and mu_star to cfg.output_path."""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        states = np.array([o.states[0, :] for o in self.orbits])
        periods = np.array([o.period for o in self.orbits])

        np.savez(
            self.cfg.output_path,
            states=states,
            periods=periods,
            mu_star=self.mu_star,
        )
        print(f"Saved {len(self.orbits)} orbits to {self.cfg.output_path}")

    def __repr__(self):
        return (
            f"Family(system={self.cfg.system.name}, "
            f"ic={self.cfg.ic.name}, "
            f"orbits={len(self.orbits)})"
        )
