from .config import RunConfig, ContinuationConfig
from .presets import PRESETS
from pathlib import Path
from .system_config import SE_SYSTEM, EM_SYSTEM

PROJECT_ROOT = Path(__file__).resolve().parent.parent


ACTIVE_CONFIG = RunConfig(
    system=EM_SYSTEM,
    ic=PRESETS["EM_L2_Northern_Halo"],
    continuation=ContinuationConfig(
        max_iter=50,
        eps=1e-6,
        step=0.01,
        max_solutions=10,
        max_period_days=30.0,
    ),
    output_dir=PROJECT_ROOT / "results",
    show_plots=True,
    spice_kernel=PROJECT_ROOT / "fullForce.txt",
)
