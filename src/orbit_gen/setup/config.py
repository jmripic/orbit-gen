from dataclasses import dataclass, field
from .orbit_ic import OrbitIC
from .system_config import SystemConfig, EM_SYSTEM, SE_SYSTEM
from .presets import PRESETS
from .kernels import ensure_kernels
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
            Maximum allowable full orbit period, in days
        max_solutions (int):
            Maximum number of converged periodic orbits to generate before
            terminating continuation
        step (float):
            Pseudo-arclength continuation step size in canonical units

    """

    max_iter: int = 50
    eps: float = 1e-6
    step: float = 0.01
    max_solutions: int = 10
    max_period_days: float = 30.0


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

    """

    system: SystemConfig
    ic: OrbitIC
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    output_dir: Path = PROJECT_ROOT.parent.parent / "results"
    show_plots: bool = True
    kernel_paths: dict[str, Path] = field(
        init=False, default_factory=ensure_kernels, repr=False
    )

    @property
    def output_filename(self) -> str:
        return f"{self.ic.name}.npz"

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RunConfig":
        with open(yaml_path) as yp:
            data = yaml.safe_load(yp)

        system = _SYSTEMS[data["system"]]
        ic = PRESETS[data["preset"]]

        continuation = ContinuationConfig(
            max_iter=data.get("max_iter", 50),
            eps=data.get("eps", 1e-6),
            step=data.get("step", 0.01),
            max_solutions=data.get("max_solutions", 10),
            max_period_days=data.get("max_period_days", 30.0),
        )

        return cls(
            system=system,
            ic=ic,
            continuation=continuation,
            output_dir=PROJECT_ROOT.parent.parent
            / Path(data.get("output_dir", "results")),
            show_plots=data.get("show_plots", True),
        )
