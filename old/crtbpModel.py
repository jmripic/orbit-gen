import numpy as np
import sys
from astropy.time import Time
import astropy.units as u
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib import animation

sys.path.insert(1, "tools")
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import extractTools
import spiceypy as spice
import multiShooting as ms

from scipy.optimize import fsolve

spice.furnsh("fullForce.txt")

# ** USER INPUTS
showPlots = True
fileDir = "results"
fileName = "L1_NorthernN.npz"

# Parameters
gmEarth = spice.bodvrd("Earth", "GM", 1)[1][0]
gmMoon = spice.bodvrd("Moon", "GM", 1)[1][0]
mu_star = gmMoon / (gmEarth + gmMoon)
# mu_star = 0.01215059
# mu_star = 1.2150568E-2
m1 = 1 - mu_star
m2 = mu_star

radiiMoon = spice.bodvrd("Moon", "RADII", 3)[1][0]
rMoon = unitConversion.convertPos_to_canonical(radiiMoon * u.km)

# Initial condition in canonical units in rotating frame R [pos, vel]
# IC = [
#     1.0110350588,
#     0,
#     -0.1731500000,
#     0,
#     -0.0780141199,
#     0,
#     1.3632096570 / 2,
# ]  # L2 Northern
# IC = [0.583856747, 0.0, 0.0, 0.0, 0.96455414, 0.0, 5.70245716 / 2]  # DRO


# IC = [0.429519110229904, 0, 0, 0, 1.440796689672539, 0, 3.051133070334277]  # DRO

# IC = [
#     0.856382122325864,
#     0,
#     0.181519309916197,
#     0,
#     0.257898218422393,
#     0,
#     1.22727308466325,
# ]  # L1

# IC = [
#     1.06896234204296,
#     0,
#     0.159599443574046,
#     0,
#     -0.00769167653854165,
#     0,
#     1.66142030228280,
# ]  # butterfly

# IC = [0.95571113, 0.0, 0.16892834, 0.0, 0.29101955, 0.0, 6.8828406 / 2]


# Generate new ICs using the free variable and constraint method
arrayI = np.reshape(np.eye(6), (1, 36))[0]
X = [IC[0], IC[2], IC[4], IC[6]]
max_iter = 50
error = 10
eps = 1e-6
step = 0.01
Tp_lim = unitConversion.convertTime_to_canonical(30.0 * u.d)
# Tp_max = 5.8
goodSols = np.array([])
Nsols = -1
ax1 = plt.figure().add_subplot(projection="3d")
while X[-1] * 2 < Tp_lim and Nsols < 10:
    ctr = 0
    error = 10
    z = np.array([0, 0, 0, -1])
    while error > eps and ctr < max_iter:
        Xfull = np.append(X, arrayI).tolist()

        Fx, Phi = orbitEOMProp.calcFx_R(Xfull, mu_star)

        error = np.linalg.norm(Fx)
        if error < eps:
            print("Error is: " + str(error))
            break

        dFx = orbitEOMProp.calcdFx_CRTBP(X, mu_star, m1, m2, Phi)

        X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

        ctr = ctr + 1
        print("Error is: " + str(error))

    if X[-1] < 0:
        break

    if error > eps:
        break

    print("Number of attempts: " + str(ctr))
    Nsols = Nsols + 1

    print(
        "Orbit period: "
        + str(unitConversion.convertTime_to_dim(2 * X[-1]).to_value(u.d))
    )

    # Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
    freeVar0CRTBP_R = X.copy()
    freeVar0CRTBP_R[-1] = 2 * freeVar0CRTBP_R[-1]
    statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(
        freeVar0CRTBP_R, mu_star
    )  # State is in the R frame
    posCRTBP_R = statesCRTBP_R[:, 0:3]
    velCRTBP_R = statesCRTBP_R[:, 3:6]

    rmag = np.linalg.norm(posCRTBP_R, axis=1)
    print(
        "Perilune: " + str(unitConversion.convertPos_to_dim(min(rmag)).to_value(u.km))
    )
    if np.any(rmag < rMoon):
        print("Intersects moon. Not a solution")
        Nsols = Nsols - 1
    else:
        ax1.plot(posCRTBP_R[:, 0], posCRTBP_R[:, 1], posCRTBP_R[:, 2])
        if showPlots:
            plt.show(block=False)
            plt.pause(0.1)

        sol0 = np.append(statesCRTBP_R[0, :], timesCRTBP_R[-1])
        goodSols = np.append(goodSols, sol0)

    # Generate new z and X for another orbit
    solp = X + z * step
    fss = fsolve(
        orbitEOMProp.fsolve_eqns,
        X,
        args=(z, solp, mu_star),
        full_output=True,
        xtol=1e-6,
    )
    X = fss[0]
    Q = fss[1]["fjac"]
    Rs = fss[1]["r"]
    R = np.zeros((4, 4))
    idx, col = np.triu_indices(4, k=0)
    R[idx, col] = Rs
    J = Q.T @ R

    try:
        z = np.linalg.inv(J) @ z
        z = z / np.linalg.norm(z)
    except:
        print("Singular matrix. Stopping continuation")
        break
    print("Solution counter: " + str(Nsols) + "\n")

ax1.set_xlabel("X [DU]")
ax1.set_ylabel("Y [DU]")
ax1.set_zlabel("Z [DU]")
if showPlots:
    plt.show(block=False)
    plt.pause(0.1)

if goodSols.size == 0:
    raise RuntimeError("All generated solutions intersect the secondary body.")

goodSols = np.reshape(goodSols, (Nsols + 1, 7))
states = goodSols[1:, 0:6]
periods = goodSols[1:, 6]
statesR, timesR = orbitEOMProp.statePropCRTBP_R(goodSols[-1, [0, 2, 4, 6]], mu_star)

ax1 = plt.figure().add_subplot(projection="3d")
ax1.plot(statesR[:, 0], statesR[:, 1], statesR[:, 2])
ax1.set_xlabel("X [DU]")
ax1.set_ylabel("Y [DU]")
ax1.set_zlabel("Z [DU]")

# save initial conditions
np.savez(
    fileDir + "/" + fileName,
    states=states,
    periods=periods,
    mu_star=mu_star,
)
