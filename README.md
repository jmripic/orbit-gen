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

Or to install with plotting dependencies

```bash
pip install "orbit-gen[plotting]"
```

Requires Python 3.11+.

## Usage

### From a config file

```yaml
# orbit_config.yaml
system: SE                      # "EM" (Earth-Moon) or "SE" (Sun-Earth)
preset: L2_Northern_Halo

max_iter: 100                   # Maximum Newton iterations for a single differential correct attempt
eps: 1.0e-10                    # Convergence tolerance on the residual norm (corrector and continuation steps)
step: 0.01                      # Pseudo-arclength step size
max_solutions: 20               # Maximum number of continuation steps
max_period_days: .inf           # Maximum full orbit period in days (.inf for no limit)

output_dir: results
show_plots: false               # Optional; for user use only
verbose: true                   # Print progress messages (warnings always show)
```

> **Note:** `step` is the distance moved along the family's tangent direction at each step. Its sign sets the initial direction: a positive value results in an initially increasing period, a negative value results in an initially decreasing period.

> **Note:** A relative `output_dir` is resolved from the folder containing the config file.

> **Note:** All continuation parameters are optional; any left out take the defaults shown above. Write numbers in exponent form with a decimal point and a signed exponent (`1.0e-10`, not `1e-10`).

```python
from orbit_gen import Family, RunConfig

cfg = RunConfig.from_yaml("orbit_config.yaml")
fam = Family(cfg)

fam.save()
if cfg.show_plots:
    fam.plot()
```

> **Note:** Constructing `Family(cfg)` runs the continuation, but does not save or plot results on its own — call `.save()` / `.plot()` explicitly, as shown above.

### Programmatically; without a config file

```python
from orbit_gen import Family, RunConfig
from orbit_gen.setup.system_config import SE_SYSTEM
from orbit_gen.setup.presets import PRESETS

cfg = RunConfig(system=SE_SYSTEM, ic=PRESETS["L2_Northern_Halo"]["SE"])
fam = Family(cfg)
fam.save()
```

Available preset keys (see `orbit_gen.setup.presets.PRESETS`):
`L2_Northern_Halo`[`SE`,`EM`], `L2_Northern_Halo`[`SE`,`EM`], `L2_Southern_Halo`[`SE`,`EM`], `DRO`[`SE`,`EM`], `L2_Lyapunov`[`SE`,`EM`], `Northern_Butterfly`[`SE`,`EM`], `Northern_Dragonfly`[`SE`,`EM`], `3_1_Lunar_Resonant_Planar`[`EM`], `L2_Double_Loop`[`SE`].

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

> **NOTE:** Use of `uv` is recommended for contributing to the repository. Installation instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

Create a fork of the repository, clone the fork onto your local system, and run

```bash
uv sync
```

to install the package in editable mode plus dev dependencies (pytest, ruff, deptry).
uv also installs a matching Python if needed.

Optionally if plotting capabilities are desired, run

```bash
uv sync --extra plotting
```

Note that plain `uv sync` removes the plotting extra

### Tests

```bash
uv run pytest
```

Note that the first run downloads NAIF SPICE kernels into user cache, so it needs network access.

### Checks

```bash
uv run ruff check     # Lint
uv run ruff format    # Format
uv run deptry src     # Check imports
```

### Scripts

The repo includes a driver script, `scripts/generate_family.py`,
which reads `orbit_config.yaml` from the project root, runs continuation,
saves results, and shows plots if `show_plots: true` and the `plotting` optional dependencies are installed.

An example configuration file (`orbit_config.example.yaml`), is included in the repository, and `orbit_config.yaml` can be created and used by running:

```bash
cp orbit_config.example.yaml orbit_config.yaml   # edit as needed
uv run scripts/generate_family.py
```

A second script, `scripts/animate_family.py`, animates the family saved for the current `orbit_config.yaml` and writes a rotating GIF to `animations/`. It requires the `plotting` optional dependencies and a saved family, so `scripts/generate_family.py` must be run first. Settings such as frame rate and resolution can be changed at the top of the script.

```bash
uv run scripts/animate_family.py
```

## License

MIT — see [LICENSE](LICENSE).




Credits and Acknowledgements
------------------------------
Created by Jayden Ripic

The differential correction and continuation algorithms were adapted from [SIOSlab/starlift](https://github.com/SIOSlab/starlift), written by Grace Genszler
