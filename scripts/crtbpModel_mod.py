import argparse
import yaml
from pathlib import Path

from orbit_gen.configs.run_config import RunConfig, ContinuationConfig
from orbit_gen.configs.system_config import EM_SYSTEM, SE_SYSTEM
from orbit_gen.configs.presets import PRESETS
from orbit_gen.family import Family

_SYSTEMS = {
    "EM": EM_SYSTEM,
    "SE": SE_SYSTEM,
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "orbit_config.yaml"


def load_config(yaml_path: Path) -> RunConfig:
    """Construct a RunConfig object from a YAML configuration file

    Parses a YAML file containing simulation and continuation settings and
    constructs a fully initialized `RunConfig` object.

    Args:
        yaml_path (Path):
            Path to the YAML configuration file defining the simulation setup.

    Returns:
        RunConfig:
            Fully populated run configuration object derived from the YAML input.
    """

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    system = _SYSTEMS[data["system"]]
    ic = PRESETS[data["preset"]]

    continuation = ContinuationConfig(
        max_iter=data.get("max_iter", 50),
        eps=data.get("eps", 1e-6),
        step=data.get("step", 0.01),
        max_solutions=data.get("max_solutions", 10),
        max_period_days=data.get("max_period_days", 30.0),
    )

    return RunConfig(
        system=system,
        ic=ic,
        continuation=continuation,
        output_dir=Path(data.get("output_dir", "results")),
        show_plots=data.get("show_plots", True),
        spice_kernel=Path(
            data.get("spice_kernel", "../src/orbit_gen/kernels/fullForce.txt")
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="CR3BP orbit continuation")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to YAML config file (default: orbit_config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    family = Family(cfg)

    family.save()

    if cfg.show_plots:
        family.plot()


if __name__ == "__main__":
    main()
