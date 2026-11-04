from pathlib import Path

from orbit_gen.family import Family
from orbit_gen.setup.config import RunConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "orbit_config.yaml"

cfg = RunConfig.from_yaml(CONFIG_PATH)
family = Family(cfg)

family.save()

if cfg.show_plots:
    family.plot()
