import pytest

from orbit_gen.setup.config import ContinuationConfig, RunConfig
from orbit_gen.setup.presets import PRESETS


def test_from_yaml_missing_keys_use_dataclass_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text('system: "EM"\npreset: "DRO"\n')
    cfg = RunConfig.from_yaml(path)
    assert cfg.continuation == ContinuationConfig()


def test_from_yaml_reads_given_keys(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        'system: "SE"\npreset: "L2_Lyapunov"\n'
        "max_iter: 7\neps: 1.0e-9\nstep: -0.02\nmax_solutions: 3\n"
        "max_period_days: .inf\n"
    )
    cont = RunConfig.from_yaml(path).continuation
    assert cont == ContinuationConfig(
        max_iter=7, eps=1e-9, step=-0.02, max_solutions=3, max_period_days=float("inf")
    )


def test_from_yaml_preset_validation(tmp_path):
    path = tmp_path / "cfg.yaml"

    # Case 1: Preset does not exist
    path.write_text('system: "SE"\npreset: "Invalid_Preset"\n')
    with pytest.raises(ValueError, match="Unknown preset"):
        RunConfig.from_yaml(path)

    # Case 2: Preset exists, but not for the selected system
    path.write_text('system: "SE"\npreset: "3_1_Lunar_Resonant_Planar"\n')
    with pytest.raises(ValueError, match="not available for system"):
        RunConfig.from_yaml(path)

    # Case 3: Both preset and system are valid
    path.write_text('system: "EM"\npreset: "3_1_Lunar_Resonant_Planar"\n')
    cfg = RunConfig.from_yaml(path)
    assert cfg.ic == PRESETS["3_1_Lunar_Resonant_Planar"]["EM"]
