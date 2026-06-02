import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
#import rebound
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
#import orbitGenCR3BP as orgen
sys.path.insert(1, '/Users/gracegenszler/Documents/Research/starlift/tools')
import unitConversion
import frameConversion


import pdb

# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10**22 kg
# m_earth = 5.97219x10**24 kg

#"Earth": 399,
#"Sun": 10,
#"Moon": 301,
#"EM Barycenter": 3
#"Solar System Barycenter": 0

def CRTBP_EOM(t,w,mu_star):
    """Equations of motion for the CRTBP in the rotating frame

    Args:
        w (float):
            State in non dimensional units [position, velocity]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """
    [x,y,z,vx,vy,vz] = w
    
    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O_H = r1*np.array([-1,0,0])
    r_2O_H = r2*np.array([1,0,0])
    
    C_I2R = frameConversion.rot(t,3)
    C_R2I = C_I2R.T
    
    r_1O_I = C_R2I@r_1O_H
    r_2O_I = C_R2I@r_2O_H

    r_PO_I = np.array([x,y,z])
    v_PO_I = np.array([vx,vy,vz])

    r_P1_I = r_PO_I - r_1O_I
    r_P2_I = r_PO_I - r_2O_I
    
    r1_mag = np.linalg.norm(r_P1_I)
    r2_mag = np.linalg.norm(r_P2_I)

    F_g1 = -m1/r1_mag**3*r_P1_I
    F_g2 = -m2/r2_mag**3*r_P2_I
    F_g = F_g1 + F_g2

    e3_hat = np.array([0,0,1])

    a_PO_H = F_g #- 2*np.cross(e3_hat,v_PO_H) - np.cross(e3_hat,np.cross(e3_hat,r_PO_H))
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]
    return dw
    
def FF_EOM(tt,w,t_mjd,mu_star,C_G2I):
    """Equations of motion for the full force model in the inertial frame

    Args:
        w (float):
            State in non dimensional units [position, velocity]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """
    
    [x,y,z,vx,vy,vz] = w
    
    gmSun = const.GM_sun.to('AU3/d2').value        # in AU^3/d^2
    gmEarth = const.GM_earth.to('AU3/d2').value
    gmMoon = 0.109318945437743700E-10              # from de432s header

    r_PEM = np.array([x,y,z])
    v_PEM = np.array([vx,vy,vz])

    time = tt + t_mjd

    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU')
    r_EarthO = get_body_barycentric_posvel('Earth',time)[0].get_xyz().to('AU')

    r_PSun = r_PEM - r_SunO.value
    r_PEarth = r_PEM - r_EarthO.value
    r_PMoon = r_PEM - r_MoonO.value
    rSun_mag = np.linalg.norm(r_PSun)
    rEarth_mag = np.linalg.norm(r_PEarth)
    rMoon_mag = np.linalg.norm(r_PMoon)

    F_gSun = -gmSun/rSun_mag**3*r_PSun
    F_gEarth = -gmEarth/rEarth_mag**3*r_PEarth
    F_gMoon = -gmMoon/rMoon_mag**3*r_PMoon
    F_g = F_gSun + F_gEarth + F_gMoon
    
    a_PO_H = F_g
    
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]

    return dw


def statePropCRTBP(freeVar,mu_star):
    """Propagates the dynamics using the free variables

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    T = freeVar[-1]

    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,)
    states = sol_int.y.T
    
    return states
    
def statePropFF(freeVar,t_mjd,mu_star,C_G2I):
    """Propagates the dynamics using the free variables

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
#    x0 = [freeVar[0].value, 0, freeVar[1].value, 0, freeVar[2].value, 0]
    T = freeVar[-1]

    sol_int = solve_ivp(FF_EOM, [0, T], freeVar[0:6], args=(t_mjd,mu_star,C_G2I), method='LSODA')

    states = sol_int.y.T
    times = sol_int.t
    
    return states, times
    
def calcMonodromyMatrix(freeVar,mu_star,m1,m2):
    """Calculates the monodromy matrix

    Args:
        pos (float):
            Position in non dimensional units
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non dimensional units
        m2 (float):
            Mass of the smaller primary in non dimensional units

    Returns:
        ~numpy.ndarray(float):
            monodromy matrix

    """
    x = freeVar[0]
    y = freeVar[1]
    z = freeVar[2]

    r_P0 = np.array([x, y, z])
    r_m10 = -np.array([mu_star, 0, 0])
    r_m20 = np.array([(1 - mu_star), 0, 0])
    
    r_Pm1 = r_P0 - r_m10
    r_Pm2 = r_P0 - r_m20

    rp1 = np.linalg.norm(r_Pm1)
    rp2 = np.linalg.norm(r_Pm2)

    dxdx = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*(x + m2)**2/rp1**5 + 3*m2*(x - 1 + m2)**2/rp2**5
    dxdy = 3*m1*(x + m2)*y/rp1**5 + 3*m2*(x - 1 + m2)*y/rp2**5
    dxdz = 3*m1*(x + m2)*z/rp1**5 + 3*m2*(x - 1 + m2)*z/rp2**5
    dydy = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*y**2/rp1**5 + 3*m2*y**2/rp2**5
    dydz = 3*m1*y*z/rp1**5 + 3*m2*y*z/rp2**5
    dzdz = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*z**2/rp1**5 + 3*m2*z**2/rp2**5

    Z = np.zeros([3,3])
    I = np.identity(3)
    A = np.array([[dxdx, dxdy, dxdz],
            [dxdy, dydy, dydz],
            [dxdz, dydz, dzdz]])
    Phi = np.block([[Z, I],
           [A, Z]])
    return Phi

def calcdFx(freeVar,mu_star,m1,m2):
    """Calculates the jacobian of the free variables wrt the constraints

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    
    s_T = statePropCRTBP(freeVar,mu_star)
    state = s_T[-1]
    
    Phi = calcMonodromyMatrix(state, mu_star,m1,m2)

    phis = np.array([[Phi[1,0], Phi[1,2], Phi[1,4]],
                    [Phi[3,0], Phi[3,2], Phi[3,4]],
                    [Phi[5,0], Phi[5,2], Phi[5,4]]])

    X = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    dw = CRTBP_EOM(freeVar[-1], X, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])
    dFx = np.zeros([3,4])
    dFx[:,0:3] = phis
    dFx[:,3] = ddT
    return dFx

def calcFx(freeVar):
    """Applies constraints to the free variables

    Args:
        freeVar (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    s_T = statePropCRTBP(freeVar,mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx

def fsolve_eqns(w,z,solp):
    Fx = calcFx(w)
    zeq = z.T@(w-solp)
    sys_w = np.append(Fx,zeq)

    return sys_w
    
def fsolve_L2(w,r1,r2):
    w_eq = w - r2*(w + r1)/np.abs(w + r1)**3 + r1*(w - r2)/np.abs(w - r2)**3

    return w_eq

#Barycentric (ICRS)
t_mjd = Time(60380,format='mjd',scale='utc')

coord.solar_system.solar_system_ephemeris.set('de432s')

r_earthO = get_body_barycentric_posvel('Earth',t_mjd)[0].get_xyz()
r_baryO = get_body_barycentric_posvel('Earth-Moon-Barycenter',t_mjd)[0].get_xyz()

mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)
X = [IC[0], IC[2], vI[1], IC[6]]

#path_str ="$HOME/Documents/Research/starlift/orbitFiles/L1_S_10.003_days.p"   # change this
#path_f1 = os.path.normpath(os.path.expandvars(path_str))
#f1 = open(path_f1, "rb")
#tmp = pickle.load(f1)
#f1.close()
#
#states = tmp['state']
#posCRTBP = (states[:,0:3]*u.km).to('AU')
#
#dim_state0 = tmp['state'][0]
#dim_Tp = tmp['te'][0][0]/2

#x_can = unitConversion.convertPos_to_canonical(dim_state0[0]*u.km)
#z_can = unitConversion.convertPos_to_canonical(dim_state0[2]*u.km)
#dy_can = unitConversion.convertVel_to_canonical(dim_state0[4]*u.km/u.s)
#Tp_can = unitConversion.convertTime_to_canonical(dim_Tp*u.s)
#IC = [x_can, 0, z_can, 0, dy_can, 0, Tp_can]
#
#X = [IC[0], IC[2], vI[1], IC[6]]
#
eps = 3E-6
N = 5
solutions = np.zeros([N,4])
zT = np.array([0, 0, 0, 1])
z = np.array([0, 0, 0, 1])
step = 1E-2

error = 10
#ctr = 0
#while error > eps:
#    Fx = calcFx(X)
#
#    error_new = np.linalg.norm(Fx)
#
#    if error_new > error:
#        print('Solution Did Not Converge')
#        print(error_new)
#        break
#    else:
#        error = error_new
#        ctr = ctr + 1
#
#    dFx = calcdFx(X,mu_star,m1,m2)
#
#    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
#
#IV = np.array([X[0], X[1], X[2], 2*X[3]])
##solutions[ii] = IV
#statesCRTBP = statePropCRTBP(IV,mu_star)

#posCRTBP = unitConversion.convertPos_to_dim(statesCRTBP[:,0:3]).to('AU').value

vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)
v_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',t_mjd)[1].get_xyz().to('AU/day')
    
x_dim = unitConversion.convertPos_to_dim(X[0]).to('AU').value
z_dim = unitConversion.convertPos_to_dim(X[1]).to('AU').value
v_dim = unitConversion.convertVel_to_dim(vI).to('AU/day')
Tp_dim = unitConversion.convertTime_to_dim(X[3]).to('day').value

pos_dim = np.array([x_dim, 0, z_dim])*u.AU

C_I2G = frameConversion.inert2geo(t_mjd)
C_G2I = C_I2G.T

pos_GCRS = C_I2G@pos_dim
pos_ICRS = (frameConversion.gmec2icrs(pos_GCRS,t_mjd)).to('AU').value

vel_ICRS = (v_EMO + v_dim).value

state0 = np.array([pos_ICRS[0], pos_ICRS[1], pos_ICRS[2], vel_ICRS[0], vel_ICRS[1], vel_ICRS[2], 6])#124*Tp_dim

statesFF, timesFF = statePropFF(state0, t_mjd, mu_star, C_G2I)
posFF = statesFF[:, 0:3]

times = np.linspace(0,366,366*2)+t_mjd
#times = np.linspace(0,6,66)+t_mjd

r_orbit1 = np.zeros([len(timesFF),3])
r_orbit2 = np.zeros([len(timesFF),3])
r_orbit3 = np.zeros([len(timesFF),3])

C_G2I = C_I2G.T
for ii in np.arange(len(timesFF)):
    time = timesFF[ii] + t_mjd
#    breakpoint()
    pos_GCRS = (frameConversion.icrs2gmec(posFF[ii]*u.AU,time)).to('AU')
        
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().to('AU')

    r_PG = pos_GCRS - r_EMO
    
    C_I2R = frameConversion.inert2rot(time,t_mjd)

    r_orbit1[ii,:] = r_PG.to('AU')
    r_orbit2[ii,:] = C_G2I@r_PG.to('AU')
    r_orbit3[ii,:] = C_G2I@C_I2R@r_PG.to('AU')
#


r_SunEM_r2 = np.zeros([len(times),3])
r_SunEM_r = np.zeros([len(times),3])
r_EarthEM_r = np.zeros([len(times),3])
r_MoonEM_r = np.zeros([len(times),3])
#r_L2EM = np.zeros([len(times),3])
#L2EM = np.zeros([len(times),1])
#L2EM[0] = 0.003
#sFlag = np.zeros([len(times),1])
for ii in np.arange(len(times)):
    time = times[ii]
    
    r_SunO = get_body_barycentric_posvel('Sun',time)[0].get_xyz().to('AU').value

    r_MoonO = get_body_barycentric_posvel('Moon',time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter',time)
    r_EMO = EMO[0].get_xyz().to('AU').value

    r_SunG = frameConversion.icrs2gmec(r_SunO*u.AU,time)
    r_MoonG = frameConversion.icrs2gmec(r_MoonO*u.AU,time)
    r_EMG = frameConversion.icrs2gmec(r_EMO*u.AU,time)

#    breakpoint()
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG
    
    C_I2G2 = frameConversion.inert2geo(time)
    C_G2I2 = C_I2G2.T
    
    C_I2R = frameConversion.inert2rot(time,t_mjd)
#    
##    print(C_I2R)
##    breakpoint()
#    
#    r_SunEM_r[ii,:] = C_G2I@r_SunEM.to('AU')
#    r_SunEM_r2[ii,:] = C_G2I2@r_SunEM.to('AU')
#    r_EarthEM_r[ii,:] = C_G2I@r_EarthEM.to('AU')
#    r_MoonEM_r[ii,:] = C_G2I@r_MoonEM.to('AU')
#    
##    breakpoint()
#
    r_SunEM_r[ii,:] = C_G2I@C_I2R@r_SunEM.to('AU')
    r_SunEM_r2[ii,:] = C_G2I2@C_I2R@r_SunEM.to('AU')
    r_EarthEM_r[ii,:] = C_G2I@C_I2R@r_EarthEM.to('AU')
    r_MoonEM_r[ii,:] = C_G2I@C_I2R@r_MoonEM.to('AU')
#    
#
##    breakpoint()
#    r_SunEM_r[ii,:] = r_SunEM.to('AU')
#    r_EarthEM_r[ii,:] = r_EarthEM.to('AU')
#    r_MoonEM_r[ii,:] = r_MoonEM.to('AU')
#
#    r1 = np.linalg.norm(r_EarthEM_r[ii,:])
#    r2 = np.linalg.norm(r_MoonEM_r[ii,:])
#    ss = fsolve(fsolve_L2,L2EM[ii],args=(r1,r2),full_output=True,xtol=1E-12)
#    L2EM[ii] = ss[0][0]
#    sFlag[ii] = ss[2]
#
#    if ss[2] > 1:
#        breakpoint()
#    r_L2 = r_MoonEM_r[ii,:] - r_EarthEM_r[ii,:]
#    r_L2hat = r_L2/np.linalg.norm(r_L2)
#    
#    r_L2EM[ii,:] = L2EM[ii]*r_L2hat
    


#p2 = unitConversion.convertPos_to_dim(1-mu_star).to('AU')
#p1 = unitConversion.convertPos_to_dim(-mu_star).to('AU')
#
#r_MoonEM = np.array([p2.value, 0, 0])
#r_EarthEM = np.array([p1.value, 0, 0])
#    
#for ii in np.arange(len(times)):
#    time = times[ii]
#
#    C_I2R = frameConversion.inert2rot(time,t_mjd)
#    C_R2I = C_I2R.T
#    C_I2G, C_LAAN, C_INC, C_AOP, n_LAAN, n_INC, n_AOP = frameConversion.inert2geo(time,t_mjd)
#    
#    r_EarthEM_r[ii,:] = r_EarthEM
#    r_MoonEM_r[ii,:] = r_MoonEM
#
##    r_EarthEM_r[ii,:] = C_R2I@r_EarthEM
##    r_MoonEM_r[ii,:] = C_R2I@r_MoonEM
##
#    r_EarthEM_r[ii,:] = C_R2I@C_I2G@r_EarthEM
#    r_MoonEM_r[ii,:] = C_R2I@C_I2G@r_MoonEM


ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_orbit1[:,0],r_orbit1[:,1],r_orbit1[:,2],'b')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')

ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_orbit2[:,0],r_orbit2[:,1],r_orbit2[:,2],'b')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')

ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_orbit3[:,0],r_orbit3[:,1],r_orbit3[:,2],'b')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(r_SunEM_r[0,0],r_SunEM_r[0,1],r_SunEM_r[0,2])
ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2],'y')
ax.plot(r_SunEM_r2[:,0],r_SunEM_r2[:,1],r_SunEM_r2[:,2],'b')
#ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'m')
#ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'g')
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')

#ax = plt.figure().add_subplot(projection='3d')
##ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2],'r')
#ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'b')
##ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2],'y')
##ax.scatter(r_SunEM_r[0,0],r_SunEM_r[0,1],r_SunEM_r[0,2])
##ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'m')
##ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'g')
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')

#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2],'r')
#ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'b')
##ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2])
#ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'m')
#ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'g')
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')


#ax = plt.figure().add_subplot(projection='3d')
##ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2])
#ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'b')
##ax.plot(posFF[7000:8000,0],posFF[7000:8000,1],posFF[7000:8000,2],'b')
#ax.scatter(posFF[0,0], posFF[0,1], posFF[0,2])
##ax.plot(r_SunEM_r[:,0],r_SunEM_r[:,1],r_SunEM_r[:,2])
##ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'r')
##ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'g')
##ax.plot(r_MoonEM_r[:,0]-r_EarthEM_r[:,0],r_MoonEM_r[:,1]-r_EarthEM_r[:,1],r_MoonEM_r[:,2]-r_EarthEM_r[:,2],'g')
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')
#breakpoint()
#ax = plt.figure().add_subplot(projection='3d')
##ax.plot(posCRTBP[:,0],posCRTBP[:,1],posCRTBP[:,2])
##ax.plot(posFF[:,0],posFF[:,1],posFF[:,2],'k')
#ax.plot(r_MoonEM_r[:,0],r_MoonEM_r[:,1],r_MoonEM_r[:,2],'g')
#ax.plot(r_EarthEM_r[:,0],r_EarthEM_r[:,1],r_EarthEM_r[:,2],'r')
#ax.plot(r_L2EM[0:100,0],r_L2EM[0:100,1],r_L2EM[0:100,2],'b')
#ax.set_xlabel('X [AU]')
#ax.set_ylabel('Y [AU]')
#ax.set_zlabel('Z [AU]')

#plt.figure()
#plt.plot(posCRTBP[:,0],posCRTBP[:,1], 'r')
#plt.plot(posFF[:,0],posFF[:,1], 'b')
#plt.xlabel('X [AU]')
#plt.ylabel('Y [AU]')
#
#plt.figure()
#plt.plot(posCRTBP[:,0],posCRTBP[:,2], 'r')
#plt.plot(posFF[:,0],posFF[:,2], 'b')
#plt.xlabel('X [AU]')
#plt.ylabel('Z [AU]')
#
#plt.figure()
#plt.plot(posCRTBP[:,1],posCRTBP[:,2], 'r')
#plt.plot(posFF[:,1],posFF[:,2], 'b')
#plt.xlabel('Y [AU]')
#plt.ylabel('Z [AU]')

plt.show()
breakpoint()

#ax.plot(posFF[2000:,0],posFF[2000:,1],posFF[2000:,2],'b')
