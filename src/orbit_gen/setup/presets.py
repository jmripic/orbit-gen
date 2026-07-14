from .orbit_ic import OrbitIC

PRESETS: dict[str, OrbitIC] = {
    "EM_L2_Northern_Halo": OrbitIC(
        name="EM_L2_Northern",
        x0=1.0110350588,
        z0=-0.1731500000,
        vy0=-0.0780141199,
        half_period=1.3632096570 / 2,
    ),
    "EM_DRO_1": OrbitIC(
        name="EM_DRO_1",
        x0=0.583856747,
        z0=0.0,
        vy0=0.96455414,
        half_period=5.70245716 / 2,
    ),
    "EM_DRO_2": OrbitIC(
        name="EM_DRO_2",
        x0=0.429519110229904,
        z0=0.0,
        vy0=1.440796689672539,
        half_period=3.051133070334277,
    ),
    "EM_L1": OrbitIC(
        name="EM_L1",
        x0=0.856382122325864,
        z0=0.181519309916197,
        vy0=0.257898218422393,
        half_period=1.22727308466325,
    ),
    "EM_Butterfly": OrbitIC(
        name="EM_Butterfly",
        x0=1.06896234204296,
        z0=0.159599443574046,
        vy0=-0.00769167653854165,
        half_period=1.66142030228280,
    ),
    "EM_Unnamed_1": OrbitIC(
        name="EM_Unnamed_1",
        x0=0.95571113,
        z0=0.16892834,
        vy0=0.29101955,
        half_period=6.8828406 / 2,
    ),
    "SE_L2_Northern_Halo": OrbitIC(
        name="SE_L2_Northern_Halo",
        x0=1.0112461245,
        z0=0.0013915124,
        vy0=-0.0091873264,
        half_period=3.064 / 2,
    ),
}
