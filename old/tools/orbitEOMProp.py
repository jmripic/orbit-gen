import numpy as np
from scipy.integrate import solve_ivp
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import astropy.constants as const
import astropy.units as u
from tools import frameConversion
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def CRTBP_EOM_R(t, w, mu_star):
    """Equations of motion for the CRTBP in the rotating frame

    Args:
        w (~numpy.ndarray(float)):
            State in non dimensional units [position, velocity]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """

    [x, y, z, vx, vy, vz] = w[:6]

    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O_H = r1 * np.array([-1, 0, 0])
    r_2O_H = r2 * np.array([1, 0, 0])

    r_PO_H = np.array([x, y, z])
    v_PO_H = np.array([vx, vy, vz])

    r_P1_H = r_PO_H - r_1O_H
    r_P2_H = r_PO_H - r_2O_H
    r1_mag = np.linalg.norm(r_P1_H)
    r2_mag = np.linalg.norm(r_P2_H)

    F_g1 = -m1 / r1_mag**3 * r_P1_H
    F_g2 = -m2 / r2_mag**3 * r_P2_H
    F_g = F_g1 + F_g2

    e3_hat = np.array([0, 0, 1])

    a_PO_H = (
        F_g - 2 * np.cross(e3_hat, v_PO_H) - np.cross(e3_hat, np.cross(e3_hat, r_PO_H))
    )
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    if len(w) > 7:
        dxdx = (
            1
            - m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * (x + m2) ** 2 / r1_mag**5
            + 3 * m2 * (x - 1 + m2) ** 2 / r2_mag**5
        )
        dxdy = 3 * m1 * (x + m2) * y / r1_mag**5 + 3 * m2 * (x - 1 + m2) * y / r2_mag**5
        dxdz = 3 * m1 * (x + m2) * z / r1_mag**5 + 3 * m2 * (x - 1 + m2) * z / r2_mag**5
        dydy = (
            1
            - m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * y**2 / r1_mag**5
            + 3 * m2 * y**2 / r2_mag**5
        )
        dydz = 3 * m1 * y * z / r1_mag**5 + 3 * m2 * y * z / r2_mag**5
        dzdz = (
            1
            - m1 / r1_mag**3
            - m2 / r2_mag**3
            + 3 * m1 * z**2 / r1_mag**5
            + 3 * m2 * z**2 / r2_mag**5
        )

        Phi = np.reshape(w[6:], (6, 6))
        Z = np.zeros([3, 3])
        I = np.identity(3)
        A = np.array([[dxdx, dxdy, dxdz], [dxdy, dydy, dydz], [dxdz, dydz, dzdz]])
        J = np.block([[Z, I], [A, Z]])

        dPhi = J @ Phi
        dPhi = np.reshape(dPhi, (1, 36))[0]

        dw = np.append([vx, vy, vz, ax, ay, az], dPhi)
    else:
        dw = [vx, vy, vz, ax, ay, az]
    return dw


def statePropCRTBP_R(freeVar, mu_star):
    """Propagates the dynamics using the free variables in the rotating frame

    Args:
        freeVar (float np.array):
            Free variable in non-dimensional units of the form [x y z dx dy dz T/2]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        tuple:
        states ~numpy.ndarray(float):
            Positions and velocities in non-dimensional units
        times ~numpy.ndarray(float):
            Times in non-dimensional units

    """

    if len(freeVar) > 7:
        x0 = np.append(
            [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0], freeVar[4:]
        ).tolist()
    else:
        x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]

    T = freeVar[3]

    sol_int = solve_ivp(
        CRTBP_EOM_R,
        [0, T],
        x0,
        args=(mu_star,),
        rtol=1e-12,
        atol=1e-12,
    )
    states = sol_int.y.T
    times = sol_int.t

    return states, times


def calcFx_R(freeVar, mu_star):
    """Applies constraints to the free variables in the CRTBP rotating frame

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            constraint array

    """

    s_T, times = statePropCRTBP_R(freeVar, mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])

    if len(state) > 7:
        Phi = state[6:]
        Phi = np.reshape(Phi, (6, 6))

        return Fx, Phi
    else:
        return Fx


def calcdFx_CRTBP(freeVar, mu_star, m1, m2, Phi):
    """Calculates the Jacobian of the free variables wrt the constraints for the CRTBP

    Args:
        freeVar (~numpy.ndarray(float)):
            x and z positions, y velocity, and half the orbit period
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non-dimensional units
        m2 (float):
            Mass of the smaller primary in non-dimensional units

    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """

    # import pdb
    # import sys

    # pdb.Pdb(stdout=sys.__stdout__).set_trace()

    phis = np.array(
        [
            [Phi[1, 0], Phi[1, 2], Phi[1, 4]],
            [Phi[3, 0], Phi[3, 2], Phi[3, 4]],
            [Phi[5, 0], Phi[5, 2], Phi[5, 4]],
        ]
    )

    X = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    dw = CRTBP_EOM_R(freeVar[-1], X, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])
    dFx = np.zeros([3, 4])
    dFx[:, 0:3] = phis
    dFx[:, 3] = ddT

    return dFx


def fsolve_eqns(w, z, solp, mu_star):
    """Finds the initial guess for a new orbit and the Jacobian for continuation

    Args:
        w (~numpy.ndarray(float)):
            symbolic representation free variables solution
        z (~numpy.ndarray(float)):
            tangent vector to move along to find corrected free variables
        solp (~numpy.ndarray(float)):
            next free variables prediction
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            system of equations as a function of w

    """

    Fx = calcFx_R(w, mu_star)
    zeq = z.T @ (w - solp)
    sys_w = np.append(Fx, zeq)

    return sys_w


# def generateFamily_CRTBP(guess, mu_star, N):
#    """Generates and plots a family of orbits in the CRTBP model given a guess for the initial state.
#    Each orbit is generated over 1 orbital period.
#
#    Args:
#        guess (float array):
#            Initial guess for the orbit family in the form [x y z dx dy dz T/2], where T is the orbit period
#        mu_star (float):
#            Non-dimensional mass parameter
#        N (float):
#            Number of orbits in the family to be generated
#
#    Returns:
#        ICs (float n array):
#            A matrix consisting of the initial states of for all the orbits in the family
#
#    """
#
#    # Parameters
#    m1 = (1 - mu_star)
#    m2 = mu_star
#
#    # Initial guess for the free variable vector
#    X = [guess[0], guess[2], guess[4], guess[6]]
#
#    eps = 1E-6
#    solutions = np.zeros([N, 4])
#    z = np.array([0, 0, 0, 1])
#    step = 1E-2
#
#    max_iter = 1000
#    ax = plt.figure().add_subplot(projection='3d')
#    ICs = np.zeros([N, 7])
#    for ii in np.arange(N):
#        error = 10
#        ctr = 0
#        while error > eps and ctr < max_iter:
#            # Generate the free variable vector
#            Fx = calcFx_R(X, mu_star)
#            error = np.linalg.norm(Fx)
#            dFx = calcdFx_CRTBP(X, mu_star, m1, m2)
#            X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)
#            ctr = ctr + 1
#
#        # Generate an orbit from the found free variable vector
#        freeVar = np.array([X[0], X[1], X[2], 2*X[3]])
#        # solutions[ii] = freeVar
#        # states, times = statePropCRTBP_R(freeVar, mu_star)
#
#        ICs[ii] = [X[0], 0, X[1], 0, X[2], 0, X[3]]
#
#        # # Plot the orbit
#        # ax.plot(states[:, 0], states[:, 1], states[:, 2])
#
#        # Generate new z and X for another orbit
#        solp = X + z * step
#        ss = fsolve(fsolve_eqns, X, args=(z, solp, mu_star), full_output=True, xtol=1E-12)
#        X = ss[0]
#        Q = ss[1]['fjac']
#        Rs = ss[1]['r']
#        R = np.zeros((4, 4))
#        idx, col = np.triu_indices(4, k=0)
#        R[idx, col] = Rs
#        J = Q.T @ R
#
#        z = np.linalg.inv(J) @ z
#        z = z / np.linalg.norm(z)
#
#    # plt.show()
#    return ICs


def jacobiConstCRTBPR(pos, vel, mu_star):

    r_Mbary = np.array([1 - mu_star, 0, 0])
    r_Ebary = np.array([-mu_star, 0, 0])

    KE = np.dot(vel, vel) / 2
    U1 = -(pos[0] ** 2 + pos[1] ** 2) / 2
    U2 = -(
        (1 - mu_star) / np.linalg.norm(pos - r_Ebary[0:3])
        + (mu_star) / np.linalg.norm(pos - r_Mbary)
    )

    C = KE + U1 + U2

    return C


def jacobiConstCRTBPI(pos, vel, mu_star, t):
    r_Mbary = np.array([1 - mu_star, 0, 0])
    r_Ebary = np.array([-mu_star, 0, 0])

    C_I2R = frameConversion.rot(t, 3)
    C_R2I = C_I2R.T

    r_MbaryI = C_R2I @ r_Mbary
    r_EbaryI = C_R2I @ r_Ebary

    r_scM = pos - r_MbaryI
    r_scE = pos - r_EbaryI

    e3 = np.array([0, 0, 1])

    KE = np.dot(vel, vel) / 2
    U1 = (1 - mu_star) / np.linalg.norm(r_scE) + mu_star / np.linalg.norm(r_scM)
    U2 = np.dot(vel, np.cross(e3, pos))

    C = KE - U1 - U2

    return C
