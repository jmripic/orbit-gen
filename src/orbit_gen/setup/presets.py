from .orbit_ic import OrbitIC

PRESETS: dict[str, dict[str, OrbitIC]] = {
    "L1_Northern_Halo": {
        "EM": OrbitIC(
            name="EM_L1_Northern_Halo",
            x0=0.856382122325864,
            z0=0.181519309916197,
            vy0=0.257898218422393,
            half_period=1.22727308466325,
        ),
        "SE": OrbitIC(
            name="SE_L1_Northern_Halo",
            x0=9.9342988420305367e-1,
            z0=1.2285790238546899e-2,
            vy0=1.3842107210222549e-2,
            half_period=2.2812718320995415e0 / 2,
        ),
    },
    "L2_Northern_Halo": {
        "EM": OrbitIC(
            name="EM_L2_Northern_Halo",
            x0=1.0924697204324003e0,
            z0=2.0118187175809177e-1,
            vy0=-2.0999964360980777e-1,
            half_period=2.5256026213792353e0 / 2,
        ),
        "SE": OrbitIC(
            name="SE_L2_Northern_Halo",
            x0=1.0082893998277027,
            z0=0.011434182696839030,
            vy0=-0.015179690420484332,
            half_period=2.7047531095113841 / 2,
        ),
    },
    "L2_Southern_Halo": {
        "EM": OrbitIC(
            name="EM_L2_Southern_Halo",
            x0=1.0110350588,
            z0=-0.1731500000,
            vy0=-0.0780141199,
            half_period=1.3632096570 / 2,
        ),
        "SE": OrbitIC(
            name="SE_L2_Southern_Halo",
            x0=1.0057032659079346,
            z0=-1.2409260038177099e-2,
            vy0=-1.2623398608372105e-2,
            half_period=2.1172370586539060 / 2,
        ),
    },
    "DRO": {
        "EM": OrbitIC(
            name="EM_DRO",
            x0=0.583856747,
            z0=0.0,
            vy0=0.96455414,
            half_period=5.70245716 / 2,
        ),
        "SE": OrbitIC(
            name="SE_DRO",
            x0=1.0005627381528093e0,
            z0=-9.9084312832447642e-27,
            vy0=-7.4041644955106137e-2,
            half_period=4.8020469554671263e-2 / 2,
        ),
    },
    "L2_Lyapunov": {
        "EM": OrbitIC(
            name="EM_L2_Lyapunov",
            x0=1.1213042091132026e0,
            z0=1.0696906809883904e-29,
            vy0=1.7168023931614440e-1,
            half_period=3.4134266416026278e0 / 2,
        ),
        "SE": OrbitIC(
            name="SE_L2_Lyapunov",
            x0=1.0030516289641727,
            z0=1.7613075082610678e-17,
            vy0=4.2071729820709863e-2,
            half_period=4.1379689311747248 / 2,
        ),
    },
    "Northern_Butterfly": {
        "EM": OrbitIC(
            name="EM_Northern_Butterfly",
            x0=1.06896234204296,
            z0=0.159599443574046,
            vy0=-0.00769167653854165,
            half_period=1.66142030228280,
        ),
        "SE": OrbitIC(
            name="SE_Northern_Butterfly",
            x0=9.9478400099057562e-1,
            z0=9.3307317319370944e-3,
            vy0=-2.7018843432500038e-3,
            half_period=3.6918732624381563 / 2,
        ),
    },
    "Northern_Dragonfly": {
        "EM": OrbitIC(
            name="EM_Northern_Dragonfly",
            x0=0.9942326995863632,
            z0=-0.009626001094285045,
            vy0=1.4506956173991967,
            half_period=6.897436155600134 / 2,
        ),
        "SE": OrbitIC(
            name="SE_Northern_Dragonfly",
            x0=1.0004505782707784,
            z0=-0.0065444284791597775,
            vy0=0.02533489808909745,
            half_period=5.622842389764573 / 2,
        ),
    },
    "3_1_Lunar_Resonant_Planar": {
        "EM": OrbitIC(
            name="EM_3_1_Lunar_Resonant_Planar",
            x0=2.4285786740195644e-3,
            z0=-7.6513528595122203e-57,
            vy0=1.1534396288254891e1,
            half_period=6.2649639398149954e0 / 2,
        ),
    },
    "L2_Double_Loop": {
        "SE": OrbitIC(
            name="SE_L2_Double_Loop",
            x0=1.0100474443654743,
            z0=0.009248459728745469,
            vy0=-0.020424319253302937,
            half_period=5.622943813223913 / 2,
        ),
    },
}
