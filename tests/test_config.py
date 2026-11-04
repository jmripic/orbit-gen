from orbit_gen.setup.config import ContinuationConfig, RunConfig


def test_from_yaml_missing_keys_use_dataclass_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text('system: "EM"\npreset: "DRO_1"\n')
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
