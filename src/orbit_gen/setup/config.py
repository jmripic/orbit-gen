from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .kernels import ensure_kernels
from .orbit_ic import OrbitIC
from .presets import PRESETS
from .system_config import EM_SYSTEM, SE_SYSTEM, SystemConfig

_SYSTEMS = {
    "EM": EM_SYSTEM,
    "SE": SE_SYSTEM,
}


@dataclass
class ContinuationConfig:
    """Parameters controlling differential correction and family continuation.

    Attributes:
        eps (float):
            Convergence tolerance on the constraint residual norm
        max_iter (int):
            Maximum number of Newton iterations for a single
            differential correction attempt
        max_period_days (float):
            Maximum allowable full orbit period, in days (no limit by default)
        max_solutions (int):
            Maximum number of continuation steps before terminating
        step (float):
            Pseudo-arclength continuation step size in canonical units

    """

    max_iter: int = 100
    eps: float = 1e-10
    step: float = 0.01
    max_solutions: int = 20
    max_period_days: float = float("inf")


@dataclass
class RunConfig:
    """Top-level configuration for a continuation run.

    Attributes:
        continuation (ContinuationConfig):
            Differential correction and continuation parameters
        ic (OrbitIC):
            Periodic orbit used to initialize continuation
        output_dir (Path):
            Directory where output files will be written
        output_filename (str):
            Name of the output .npz file containing continuation results
        show_plots (bool):
            If True, display plots interactively during execution
        spice_kernel (Path):
            Path to the SPICE meta-kernel furnished before propagation and
            analysis
        system (SystemConfig):
            CR3BP system used
        verbose (bool):
            Prints progress messages during continuation when true

    """

    system: SystemConfig
    ic: OrbitIC
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    output_dir: Path = Path("results")
    show_plots: bool = False
    verbose: bool = True
    kernel_paths: dict[str, Path] = field(
        init=False, default_factory=ensure_kernels, repr=False
    )

    @property
    def output_filename(self) -> str:
        """str: Output .npz filename, from the initial condition's name."""
        return f"{self.ic.name}.npz"

    @property
    def output_path(self) -> Path:
        """Path: Full path of the output .npz file, output_dir / output_filename."""
        return self.output_dir / self.output_filename

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RunConfig":
        """Build a RunConfig from a YAML file

        Args:
            yaml_path (str or pathlib.Path):
                Path to the YAML config.

        Returns:
            config (RunConfig):
                Run configuration built from the file.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path) as yp:
            data = yaml.safe_load(yp)

        system = _SYSTEMS[data["system"]]

        if data["preset"] not in PRESETS:
            raise ValueError(f"Unknown preset: {data['preset']}")
        if data["system"] not in PRESETS[data["preset"]]:
            available = list(PRESETS[data["preset"]].keys())
            raise ValueError(
                f"Preset {data['preset']} is not available for system {data['system']}. "
                f"Available systems: {available}"
            )

        ic = PRESETS[data["preset"]][data["system"]]

        names = {f.name for f in fields(ContinuationConfig)}
        continuation = ContinuationConfig(
            **{key: value for key, value in data.items() if key in names}
        )

        return cls(
            system=system,
            ic=ic,
            continuation=continuation,
            output_dir=yaml_path.parent / data.get("output_dir", "results"),
            show_plots=data.get("show_plots", False),
            verbose=data.get("verbose", True),
        )
