# orbit-gen

Periodic orbit family generation and continuation in the Circular Restricted
Three-Body Problem (CR3BP).

## Overview

`orbit-gen` generates families of periodic orbits in the CR3BP using
single-shooting differential correction and pseudo-arclength continuation.
It currently supports various families in the Earth-Moon and Sun-Earth systems.

## Installation

```bash
pip install orbit-gen
```
Requires Python 3.12+.

## Usage

### From a config file

```yaml
# orbit_config.yaml
system: SE                      # "EM" (Earth-Moon) or "SE" (Sun-Earth)
preset: SE_L2_Northern_Halo

max_iter: 1000                  # Maximum Newton iterations for a single differential correct attempt
eps: 1.0e-6                     # Differential-correction convergence tolerance on residual norm
step: 0.01                      # Pseudo-arclength continuation step size in canonical units
max_solutions: 20               # Maximum number of orbits to generate
max_period_days: 1000.0         # Maximum full orbit period in days

output_dir: results
show_plots: true                # Optional; for user use only
```

```python
from orbit_gen import Family, RunConfig

cfg = RunConfig.from_yaml("orbit_config.yaml")
fam = Family(cfg)

fam.save()
if cfg.show_plots:
    fam.plot()
```

> **Note:** constructing `Family(cfg)` runs the continuation, but does not save or plot results on its own — call `.save()` / `.plot()` explicitly, as shown above.

### Programmatically

```python
from orbit_gen import Family, RunConfig
from orbit_gen.setup.system_config import SE_SYSTEM
from orbit_gen.setup.presets import PRESETS

cfg = RunConfig(system=SE_SYSTEM, ic=PRESETS["SE_L2_Northern_Halo"])
fam = Family(cfg)
fam.save()
```

Available preset keys (see `orbit_gen.setup.presets.PRESETS`):
`EM_L2_Northern_Halo`, `EM_DRO_1`, `EM_DRO_2`, `EM_L1`, `EM_Butterfly`,
`EM_Unnamed_1`, `SE_L2_Northern_Halo`, `SE_L1_Vertical`, `SE_L2_Lyapunov`,
`SE_L2_Small_Lyapunov`, `SE_L2_Large_Lyapunov`, `SE_L2_Southern_Halo`,
`SE_Butterfly`, `SE_Dragonfly`.

## Supported Systems

| Key | System |
|-----|--------|
| `EM` | Earth-Moon |
| `SE` | Sun-Earth |

## Output

`fam.save()` writes a `.npz` file (named after the preset, in `cfg.output_dir`)
containing initial states, periods, and `mu_star` for all converged orbits:

```python
import numpy as np
data = np.load("results/SE_L2_Northern_Halo.npz")
data["states"]              # (N, 6) containing initial [x, y, z, vx, vy, vz] in canonical units for each orbit
data["periods"]             # (N,) containing full period in canonical units for each orbit
data["mu_star"]             # System mass ratio
```

## Contributing

Clone or download the repository and run

```bash
uv sync
```

to install the package in editable mode plus dev dependencies. Run the test suite with:

```bash
uv run pytest
```

The repo includes a driver script, `scripts/crtbpModel.py`,
which reads `orbit_config.yaml` from the project root, runs continuation,
saves results, and shows plots.

An example configuration file (`orbit_config.example.yaml`), is included in the repository, and `orbit_config.yaml` can be created and used by running:

```bash
cp orbit_config.example.yaml orbit_config.yaml   # edit as needed
uv run python scripts/crtbpModel.py
```

## License

MIT — see [LICENSE](LICENSE).
