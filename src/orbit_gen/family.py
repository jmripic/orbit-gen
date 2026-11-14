import warnings

import astropy.units as u
import numpy as np
import spiceypy as spice

from orbit_gen.orbit import Orbit
from orbit_gen.setup.config import RunConfig
from orbit_gen.setup.system_config import CanonicalUnits
from orbit_gen.utils import eom, units


class Family:
    """Family of CR3BP periodic orbits

    Generates a family of periodic orbits in the circular restricted
    three-body problem (CR3BP).

    Args:
        cfg (RunConfig):
            Full run configuration.

    Attributes:
        cfg (RunConfig):
            Full run configuration.
        mu_star (float):
            CR3BP mass ratio.
        m1 (float):
            Normalized primary mass (1 - mu_star).
        m2 (float):
            Normalized secondary mass (mu_star).
        r_primary (float):
            Radius of the primary body in canonical distance units (DU).
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

        dist_km = sys.canonical_units.dist_km
        time_seconds = np.sqrt(dist_km**3 / (gm_primary + gm_secondary))
        sys.canonical_units = CanonicalUnits(
            dist_km=dist_km, time_days=time_seconds / 86400
        )

        radii = spice.bodvrd(sys.primary, "RADII", 3)[1][0]
        self.r_primary = units.pos_to_canonical(radii * u.km, sys.canonical_units)
        radii = spice.bodvrd(sys.secondary, "RADII", 3)[1][0]
        self.r_secondary = units.pos_to_canonical(radii * u.km, sys.canonical_units)

    def _run_continuation(self):
        """Outer pseudo-arclength loop. Populates self.orbits."""
        cont = self.cfg.continuation
        Tp_lim = units.time_to_canonical(
            cont.max_period_days * u.d, self.cfg.system.canonical_units
        )
        stm_identity = np.reshape(np.eye(6), (1, 36))[0]
        min_period = min(
            2 * np.pi * np.sqrt(self.r_primary**3 / self.m1),
            2 * np.pi * np.sqrt(self.r_secondary**3 / self.m2),
        )

        # Free variables: [x0, z0, vy0, T/2]
        X = self.cfg.ic.as_list
        z = None
        z0_first = None

        n_steps = 0
        while X[-1] * 2 < Tp_lim and n_steps < cont.max_solutions:
            n_steps += 1
            X, error, stm_half_T = self._correct_orbit(X, stm_identity)

            if 2 * X[-1] < min_period:
                warnings.warn(
                    "Period below the shortest orbit possible outside the primary "
                    "and secondary — stopping continuation.",
                    stacklevel=1,
                )
                break
            if error > cont.eps:
                warnings.warn("Corrector did not converge.", stacklevel=1)
                break

            if abs(X[1]) < 1e-9 and abs(X[2]) < 1e-9:
                warnings.warn(
                    "Collapsed to an equilibrium point — stopping continuation.",
                    stacklevel=1,
                )
                break

            if z0_first is None:
                z0_first = X[1]
            if abs(z0_first) > 1e-6 and X[1] / z0_first < 1e-3:
                warnings.warn(
                    "Family reached a planar orbit (bifurcation) — "
                    "stopping continuation.",
                    stacklevel=1,
                )
                break

            if self.cfg.verbose:
                period_days = units.time_to_dimensional(
                    2 * X[-1], self.cfg.system.canonical_units
                ).to_value(u.day)
                print(f"Period: {period_days:.4f} days")

            states, times, valid = self._propagate_and_validate(X)

            if valid:
                orbit = Orbit(
                    x0=X[0],
                    z0=X[1],
                    vy0=X[2],
                    period=times[-1],
                    mu_star=self.mu_star,
                    canonical_units=self.cfg.system.canonical_units,
                )
                # Store already-computed trajectory so propagate() isn't needed again
                orbit.states = states
                orbit.times = times
                S = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
                orbit.monodromy = S @ np.linalg.inv(stm_half_T) @ S @ stm_half_T
                self.orbits.append(orbit)

            X, z, success = self._continuation_step(X, z)
            if not success:
                break

            if self.cfg.verbose:
                print(f"Orbits found: {len(self.orbits)}\n")

        if not self.orbits:
            raise RuntimeError("Continuation failed")

    def _correct_orbit(
        self, X: list, stm_identity: np.ndarray
    ) -> tuple[list, float, np.ndarray]:
        """Single-shooting differential corrector

        Newton-style differential correction using a single-shooting formulation
        for periodic orbit boundary condition enforcement. Steps are damped by
        0.3 while the residual is above 1e-4, where a full Newton step can
        overshoot onto a different orbit, and taken in full below it.

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
                Final residual norm ||F(X)|| at termination (inf if a
                propagation failed or the residual is not finite).
            stm_half_T (ndarray):
                State transition matrix Phi(T/2, 0) at the returned X when
                converged.
        """

        cont = self.cfg.continuation
        error = np.inf
        stm_half_T = np.eye(6)

        for i in range(cont.max_iter):
            X_full = np.append(X, stm_identity).tolist()

            try:
                Fx, state_half_T, stm_half_T = eom.constraint(
                    X_full, self.mu_star, r_min=0.1 * self.r_secondary
                )
            except RuntimeError:
                error = np.inf
                break
            assert stm_half_T is not None
            error = np.linalg.norm(Fx)

            if not np.isfinite(error):
                error = np.inf
                break
            if error < cont.eps:
                break

            dFx = eom.jacobian(X, self.mu_star, state_half_T, stm_half_T)
            step_factor = 0.3 if error > 1e-4 else 1.0
            X = X - step_factor * dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

            if self.cfg.verbose:
                print(
                    f"  iter {i + 1:2d} | error = {error:.3e} | "
                    f"cond = {np.linalg.cond(dFx @ dFx.T):.2e}"
                )

        return X, error, stm_half_T

    def _propagate_and_validate(self, X: list) -> tuple[np.ndarray, np.ndarray, bool]:
        """Propagate one full orbital period and check primary and secondary intersection

        Numerically integrates the trajectory over a full orbital period from the
        given initial conditions, then evaluates whether the trajectory intersects
        or enters the primary or secondary body.

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
                False if the trajectory intersects the primary or secondary body;
                True otherwise.
        """

        X_full = X.copy()
        X_full[-1] = 2 * X_full[-1]

        states, times = eom.propagate(X_full, self.mu_star)
        positions = states[:, 0:3]
        valid = True
        for body, center, radius in (
            ("primary", -self.mu_star, self.r_primary),
            ("secondary", 1 - self.mu_star, self.r_secondary),
        ):
            r_mag = np.linalg.norm(positions - [center, 0, 0], axis=1)

            if self.cfg.verbose:
                closest_km = units.pos_to_dimensional(
                    min(r_mag), self.cfg.system.canonical_units
                ).to_value(u.km)
                print(f"  Closest approach to the {body} body: {closest_km:.2f} km")

            if np.any(r_mag < radius):
                valid = False
                if self.cfg.verbose:
                    print(f"  Intersects {body} — discarding.")

        return states, times, valid

    def _continuation_step(
        self, X: list, z: np.ndarray | None
    ) -> tuple[list, np.ndarray, bool]:
        """Advance one pseudo-arclength continuation step along an orbit family

        Predicts the next orbit a distance `step` along the family tangent, then
        solves the periodicity constraints together with the pseudo-arclength
        condition by Newton's method with the exact Jacobian
        from the state transition matrix. If Newton does not converge within 10
        iterations, or the new orbit fails the same-family check, the step is
        halved and retried, up to 6 times. The same-family check requires the
        chord from the old to the new orbit to point roughly along the mean of the two
        family tangents.

        Args:
            X (ndarray):
                Current free-variable vector at the current solution point.
            z (ndarray or None):
                Current tangent direction vector along the continuation manifold.
                None on the first step: the family tangent at X is used, oriented
                so the period initially increases for a positive step.

        Returns:
            X_new (ndarray):
                Corrected free-variable vector for the next orbit.
            z_new (ndarray):
                Updated unit tangent direction at the new solution point.
            success (bool):
                False if no step size gave an accepted orbit; True otherwise.
        """

        cont = self.cfg.continuation
        stm_identity = np.reshape(np.eye(6), (1, 36))[0]
        X_old = np.asarray(X, dtype=float)

        if z is None:
            X_full = np.append(X_old, stm_identity).tolist()
            _, state_half_T, stm_half_T = eom.constraint(X_full, self.mu_star)
            dFx = eom.jacobian(X_old, self.mu_star, state_half_T, stm_half_T)
            null: np.ndarray = np.linalg.svd(dFx)[2][-1]
            z = null if null[3] > 0 else -null

        step = cont.step
        rejected = False
        for _ in range(7):
            sol_predictor = X_old + z * step
            X_new = sol_predictor.copy()
            error_first = None

            for _ in range(10):
                X_full = np.append(X_new, stm_identity).tolist()
                try:
                    Fx, state_half_T, stm_half_T = eom.constraint(
                        X_full, self.mu_star, r_min=0.1 * self.r_secondary
                    )
                except RuntimeError:
                    break
                G = np.append(Fx, z @ (X_new - sol_predictor))
                error = np.linalg.norm(G)
                if error_first is None:
                    error_first = error
                if not error < 1e3 * max(error_first, 1e-8):
                    break

                dFx = eom.jacobian(X_new, self.mu_star, state_half_T, stm_half_T)
                if error < cont.eps:
                    z_new = np.linalg.svd(dFx)[2][-1]
                    z_new = z_new if z_new @ z > 0 else -z_new
                    chord = (X_new - X_old) / np.linalg.norm(X_new - X_old)
                    mean = np.sign(step) * (z + z_new) / np.linalg.norm(z + z_new)
                    if np.linalg.norm(chord - mean) > 0.04:
                        rejected = True
                        break
                    return X_new.tolist(), z_new, True

                try:
                    X_new = X_new - np.linalg.solve(np.vstack([dFx, z]), G)
                except np.linalg.LinAlgError:
                    break

            step = step / 2

        if rejected:
            message = "no step size satisfied the same-family check"
        else:
            message = "corrector did not converge at any step size"
        warnings.warn(f"Continuation stopped: {message}", stacklevel=1)
        return X_old.tolist(), z, False

    def plot(self):
        """Plot all orbits in the family on a single 3D figure

        Generates a 3D visualization of all stored periodic orbits in the family,
        typically in canonical CR3BP coordinates. Each orbit is plotted in the same
        figure for qualitative comparison of geometry across the continuation family.

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

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for orbit in self.orbits:
            orbit.plot(ax=ax)

        ax.set_title(f"{self.cfg.ic.name} family — {len(self.orbits)} orbits")

        plt.show()

    def save(self):
        """Save all orbit ICs, periods and mu_star to cfg.output_path."""
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
        """Return a summary of the system, preset, and orbit count."""
        return (
            f"Family(system={self.cfg.system.name}, "
            f"ic={self.cfg.ic.name}, "
            f"orbits={len(self.orbits)})"
        )
