from .orbit_ic import OrbitIC

PRESETS: dict[str, dict[str, OrbitIC]] = {
    "L2_Southern_Halo": {
        "EM" : OrbitIC(
            name="EM_L2_Southern_Halo",
            x0=1.0110350588,
            z0=-0.1731500000,
            vy0=-0.0780141199,
            half_period=1.3632096570 / 2,
        ),
        "SE" : OrbitIC(
        name="SE_L2_Southern_Halo",
        x0=1.0057032659079346,
        z0=-1.2409260038177099e-2,
        vy0=-1.2623398608372105e-2,
        half_period=2.1172370586539060 / 2,
        ),
    },

    "DRO_1": {
        "EM" : OrbitIC(
            name="EM_DRO_1",
            x0=0.583856747,
            z0=0.0,
            vy0=0.96455414,
            half_period=5.70245716 / 2,
        ),
    },

    "L1_Northern_Halo": {
        "EM" : OrbitIC(
            name="EM_L1_Northern_Halo",
            x0=0.856382122325864,
            z0=0.181519309916197,
            vy0=0.257898218422393,
            half_period=1.22727308466325,
        ),
    },

    "Butterfly": {
        "EM" : OrbitIC(
            name="EM_Butterfly",
            x0=1.06896234204296,
            z0=0.159599443574046,
            vy0=-0.00769167653854165,
            half_period=1.66142030228280,
        ),
        "SE" : OrbitIC(
            name="SE_Butterfly",
            x0=9.9478400099057562e-1,
            z0=9.3307317319370944e-3,
            vy0=-2.7018843432500038e-3,
            half_period=3.6918732624381563 / 2,
        ),
    },

    "L2_Northern_Halo": {
        "SE" : OrbitIC(
            name="SE_L2_Northern_Halo",
            x0=1.0082893998277027,
            z0=0.011434182696839030,
            vy0=-0.015179690420484332,
            half_period=2.7047531095113841 / 2,
        ),
    },

    "L2_Lyapunov": {
        "SE" : OrbitIC(
            name="SE_L2_Lyapunov",
            x0=1.0030516289641727,
            z0=1.7613075082610678e-17,
            vy0=4.2071729820709863e-2,
            half_period=4.1379689311747248 / 2,
        ),
    },

    "Dragonfly": {
        "SE" : OrbitIC(
            name="SE_Dragonfly",
            x0=1.0087123143716323,
            z0=1.0173103235716078e-2,
            vy0=-1.6680589951751042e-2,
            half_period=5.6228423897645730 / 2,
        ),
    },
}
