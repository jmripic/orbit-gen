from orbit_gen.configs.config import RunConfig
from orbit_gen.family import Family
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "orbit_config.yaml"

cfg = RunConfig.from_yaml(CONFIG_PATH)
family = Family(cfg)

family.save()

if cfg.show_plots:
    family.plot()
