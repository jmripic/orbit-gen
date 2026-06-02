import numpy as np
import os.path
import pickle
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
#import plot_tools

#import tools.unitConversion as unitConversion
#import tools.frameConversion as frameConversion
#import tools.orbitEOMProp as orbitEOMProp
#import tools.plot_tools as plot_tools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS  ~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de432s')

# Parameters
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

C_I2G = frameConversion.inert2geo(t_start,t_veq)
C_G2I = C_I2G.T

# Initial condition in non dimensional units in rotating frame R [pos, vel]
IC = [1.0110350593505575, 0, -0.17315000084377485, 0, -0.0780142664611386, 0, 0.6816048399338378]
X = [IC[0], IC[2], IC[4], IC[6]]

max_iter = 1000
error = 10
ctr = 0
eps = 4E-6
while error > eps and ctr < max_iter:
    Fx = orbitEOMProp.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    print(error)
    dFx = orbitEOMProp.calcdFx_CRTBP(X,mu_star,m1,m2)

    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
    
    ctr = ctr + 1
    
IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])
    
# Convert the velocity to I frame from R frame
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], 1*IC[-1]])

# Propagate the dynamics in the CRTBP model
statesCRTBP, timesCRTBP = orbitEOMProp.statePropCRTBP(freeVar_CRTBP, mu_star)
posCRTBP = statesCRTBP[:, 0:3]
velCRTBP = statesCRTBP[:, 3:6]

# Preallocate space
r_PO_CRTBP = np.zeros([len(timesCRTBP), 3])
r_PEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_EarthEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_MoonEM_CRTBP = np.zeros([len(timesCRTBP), 3])
r_CRTBP_rot = np.zeros([len(timesCRTBP), 3])
r_CRTBP_I = np.zeros([len(timesCRTBP), 3])
r_CRTBP_G = np.zeros([len(timesCRTBP), 3])
r_CRTBP_G2 = np.zeros([len(timesCRTBP), 3])
r_CRTBP_I2 = np.zeros([len(timesCRTBP), 3])
r_diff = np.zeros([len(timesCRTBP), 3])

r_PEM_CRTBP_R = np.zeros([len(timesCRTBP), 3])
r_MoonEM_CRTBP_R = np.zeros([len(timesCRTBP), 3])

# sim time in mjd
times_dim = unitConversion.convertTime_to_dim(timesCRTBP)
timesCRTBP_mjd = Time(times_dim.value + t_start.value, format='mjd', scale='utc')

# Rotate CRTBP to different frames
for kk in np.arange(len(timesCRTBP_mjd)):
    time = timesCRTBP_mjd[kk]

    # Positions of the Moon and EM barycenter relative SS barycenter in H frame
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    r_EMO = EMO[0].get_xyz().to('AU').value

    # Convert from H frame to GCRS frame
    r_EMG = (frameConversion.icrs2gmec(r_EMO * u.AU, time)).to('AU')

    # Convert to AU
    r_dim = (unitConversion.convertPos_to_dim(posCRTBP[kk, :])).to('AU').value
    r_EM = C_I2G @ r_dim

    r_PO_H, _ = frameConversion.convertSC_I2H(posCRTBP[kk,:], velCRTBP[kk,:], time, C_I2G)
    r_PO_CRTBP[kk, :] = r_PO_H
    
    r_PE_GME = frameConversion.icrs2gmec(r_PO_H, time)
    r_CRTBP_G[kk, :] = r_EM
    r_CRTBP_G2[kk,:] = (r_PE_GME - r_EMG).to('AU')
#    r_CRTBP_G2[kk,:] = (r_PE_GME).to('AU')
    
    r_CRTBP_I[kk,:] = r_dim
    r_CRTBP_I2[kk, :] = C_G2I @ (r_PE_GME - r_EMG).to('AU')
    
    C_I2R = frameConversion.inert2rot(time,t_start)
    r_CRTBP_rot[kk,:] = C_I2R @ r_dim


# Convert position from I frame to H frame [AU]
pos_H, vel_H = frameConversion.convertSC_I2H(posCRTBP[0], velCRTBP[0], t_start, C_I2G)

N = 8

dt_epoch = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N)
dt_int = (timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value/(N-1)
taus = Time(np.zeros(N), format='mjd', scale='utc')
Ts = Time(np.ones(N-1)*dt_int, format='jd', scale='utc')
posvel = np.array([])
nodeInds = np.array([])
for ii in np.arange(N):
    time_i = ii*dt_epoch

    difference_array_i = np.absolute(times_dim.value-time_i)
    index_i = difference_array_i.argmin()
    nodeInds = np.append(nodeInds,index_i)

    taus[ii] = timesCRTBP_mjd[index_i]
    pos_i = posCRTBP[index_i]
    vel_i = velCRTBP[index_i]

    pos_Hi, vel_Hi = frameConversion.convertSC_I2H(pos_i, vel_i, taus[ii], C_I2G)
    posvel = np.append(posvel,np.append(pos_Hi.value, vel_Hi.value))

eps = 1E-8
error = 10
X = np.append(np.append(posvel,Ts.value),taus.value)
#print(X)
while error > eps:
    Fx = orbitEOMProp.calcFx_FF(posvel, taus, N, Ts)
    
    error = np.linalg.norm(Fx)
    print('Error is')
    print(error)
    dFx = orbitEOMProp.calcdFx_FF(posvel, taus, N, Ts)
    Xold = X
    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)
    posvel = X[0:6*N]
    Ts = Time(X[6*N:7*N-1], format='jd', scale='utc')
    taus = Time(X[7*N-1:], format='mjd', scale='utc')
    
    diff = X - Xold
    if np.any(diff > .1):
        breakpoint()

ctr = 0
posH = np.array([np.nan, np.nan, np.nan])
pos_msI = np.array([np.nan, np.nan, np.nan])
pos_msG = np.array([np.nan, np.nan, np.nan])
posR = np.array([np.nan, np.nan, np.nan])
posEM = np.array([np.nan, np.nan, np.nan])
nanArray = np.array([np.nan, np.nan, np.nan])
timesAll = np.array([])
for ii in np.arange(N-1):
    IC = np.append(X[ctr*6:((ctr+1)*6)], Ts[ctr].value)
    tau = taus[ctr]
    states, timesT = orbitEOMProp.statePropFF(IC, tau)
    posFF = states[:, 0:3]

    posH = np.block([[posH],[posFF]])
    posH = np.block([[posH],[nanArray]])
    
    tt = tau + timesT
    timesAll = np.append(timesAll, tt)
    timesAll = np.append(timesAll, Time(0, scale='utc',format='mjd'))

    ctr = ctr + 1
    
posH = posH[1:, :]

## Define the initial state array
#state0 = np.append(np.append(pos_H.value, vel_H.value), Tp_dim.value)   # Change to Tp_dim.value for one orbit
#
## Propagate the dynamics in the full force model (H frame) [AU]
#statesFF, timesFF = orbitEOMProp.statePropFF(state0, t_start)
#posFF = statesFF[:, 0:3]
#velFF = statesFF[:, 3:6]

goodInds = np.arange(len(timesAll))
#goodInds = (np.arange(0,len(timesAll),np.floor(len(timesAll)/len(timesCRTBP_mjd)))).astype(int)
timesPartial = timesAll[goodInds]
posHPartial = posH[goodInds,:]
pos_msI = np.array([np.nan,np.nan,np.nan])
pos_msG = np.array([np.nan,np.nan,np.nan])
posR = np.array([np.nan,np.nan,np.nan])
for ii in np.arange(len(timesPartial)):
    tt = timesPartial[ii]

    if tt.value  < 1:
        pos_msI = np.block([[pos_msI],[nanArray]])
        pos_msG = np.block([[pos_msG],[nanArray]])
        posEM = np.block([[posEM],[nanArray]])
        posR = np.block([[posR],[nanArray]])
    else:
        state_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', tt)
        r_EMG_icrs = state_EM[0].get_xyz().to('AU')
        
        r_PE_gcrs = frameConversion.icrs2gmec(posHPartial[ii,:]*u.AU,tt)
        r_EME_gcrs = frameConversion.icrs2gmec(r_EMG_icrs,tt)
        r_PEM = r_PE_gcrs - r_EME_gcrs

        C_I2R3 = frameConversion.inert2rot(tt,t_start)
        
        r_PEM_I = C_G2I@r_PEM
        r_PEM_r = C_I2R3@r_PEM_I

        posR = np.block([[posR],[r_PEM_r.to('AU')]])
        pos_msI = np.block([[pos_msI],[r_PEM_I.to('AU')]])
    #    pos_msGCRS = np.block([[pos_msGCRS],[r_PE_gcrs.to('AU')]])
        pos_msG = np.block([[pos_msG],[r_PEM.to('AU')]])
        posEM = np.block([[posEM],[r_EME_gcrs.to('AU')]])
    
posTMP = np.array([np.nan,np.nan,np.nan])
r_CRTBP_EMs = np.array([np.nan,np.nan,np.nan])
r_CRTBP2_I = np.array([np.nan,np.nan,np.nan])
r_CRTBP2_rot = np.array([np.nan,np.nan,np.nan])
for ii in np.arange(len(timesCRTBP)):
    tt = timesCRTBP_mjd[ii]

    state_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', tt)
    r_EMG_icrs = state_EM[0].get_xyz().to('AU')
    
    r_CRTBP_gcrs = frameConversion.icrs2gmec(r_PO_CRTBP[ii,:]*u.AU,tt)
    r_EME_gcrs = frameConversion.icrs2gmec(r_EMG_icrs,tt)
    r_CRTBP_EM = r_CRTBP_gcrs - r_EME_gcrs

    C_I2R2 = frameConversion.inert2rot(tt,t_start)
    
    tmp1 = C_G2I@r_CRTBP_EM
    tmp2 = C_I2R2@tmp1
    
    posTMP = np.block([[posTMP],[tmp1.to('AU')]])
    r_CRTBP_EMs = np.block([[r_CRTBP_EMs],[r_CRTBP_EM.to('AU')]])
    r_CRTBP2_I = np.block([[r_CRTBP2_I],[tmp1.to('AU')]])
    r_CRTBP2_rot = np.block([[r_CRTBP2_rot],[tmp2.to('AU')]])
#    posTMP = np.block([[posTMP],[r_CRTBP_gcrs.to('AU')]])

posR = posR[1:, :]
pos_msI = pos_msI[1:, :]
pos_msG = pos_msG[1:, :]
posTMP = posTMP[1:, :]
posEM = posEM[1:, :]
r_CRTBP_EMs = r_CRTBP_EMs[1:, :]
r_CRTBP2_I = r_CRTBP2_I[1:, :]
r_CRTBP2_rot = r_CRTBP2_rot[1:, :]

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(posH[:, 0], posH[:, 1], posH[:, 2], 'b', label='Multi Segment')
ax1.plot(r_PO_CRTBP[:, 0], r_PO_CRTBP[:, 1], r_PO_CRTBP[:, 2], 'r-.', label='CRTBP')
ax1.scatter(posH[0, 0], posH[0, 1], posH[0, 2])
ax1.scatter(r_PO_CRTBP[0, 0], r_PO_CRTBP[0, 1], r_PO_CRTBP[0, 2])
ax1.set_title('FF vs CRTBP in H frame (ICRS)')
ax1.set_xlabel('X [AU]')
ax1.set_ylabel('Y [AU]')
ax1.set_zlabel('Z [AU]')
plt.legend()

#fig3, ax3 = plt.subplots(2,2)
#ax3[0,1].plot(posH[:, 0], posH[:, 1], 'b')
#ax3[0,1].plot(r_PO_CRTBP[:, 0], r_PO_CRTBP[:, 1], 'r')
#ax3[0,1].set_xlabel('X [AU]')
#ax3[0,1].set_ylabel('Y [AU]')
#ax3[1,0].plot(posH[:, 0], posH[:, 2], 'b')
#ax3[1,0].plot(r_PO_CRTBP[:, 0], r_PO_CRTBP[:, 2], 'r')
#ax3[1,0].set_xlabel('X [AU]')
#ax3[1,0].set_ylabel('Z [AU]')
#ax3[1,1].plot(posH[:, 1], posH[:, 2], 'b')
#ax3[1,1].plot(r_PO_CRTBP[:, 1], r_PO_CRTBP[:, 2], 'r')
#ax3[1,1].set_xlabel('Y [AU]')
#ax3[1,1].set_ylabel('Z [AU]')

ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(pos_msG[:, 0], pos_msG[:, 1], pos_msG[:, 2], 'b', label='Multi Segment')
ax2.plot(r_CRTBP_G[:, 0], r_CRTBP_G[:, 1], r_CRTBP_G[:, 2], 'r', label='CRTBP')
#ax2.plot(posTMP[:, 0], posTMP[:, 1], posTMP[:, 2],'g',label='CRTBP w/ MS process')
ax2.scatter(pos_msG[0, 0], pos_msG[0, 1], pos_msG[0, 2])
ax2.scatter(r_CRTBP_G[0, 0], r_CRTBP_G[0, 1], r_CRTBP_G[0, 2])
ax2.set_title('FF vs CRTBP in G frame (GCRS centered at EM Barycenter)')
ax2.set_xlabel('X [AU]')
ax2.set_ylabel('Y [AU]')
ax2.set_zlabel('Z [AU]')
plt.legend()
#
#fig3, ax3 = plt.subplots(2,2)
#ax3[0,1].plot(pos_msG[:, 0], pos_msG[:, 1], 'b')
#ax3[0,1].plot(r_CRTBP_G[:, 0], r_CRTBP_G[:, 1], 'r')
#ax3[0,1].plot(posTMP[:, 0], posTMP[:, 1], 'g')
#ax3[0,1].set_xlabel('X [AU]')
#ax3[0,1].set_ylabel('Y [AU]')
#ax3[1,0].plot(pos_msG[:, 0], pos_msG[:, 2], 'b')
#ax3[1,0].plot(r_CRTBP_G[:, 0], r_CRTBP_G[:, 2], 'r')
#ax3[1,0].plot(posTMP[:, 0], posTMP[:, 2], 'g')
#ax3[1,0].set_xlabel('X [AU]')
#ax3[1,0].set_ylabel('Z [AU]')
#ax3[1,1].plot(pos_msG[:, 1], pos_msG[:, 2], 'b')
#ax3[1,1].plot(r_CRTBP_G[:, 1], r_CRTBP_G[:, 2], 'r')
#ax3[1,1].plot(posTMP[:, 1], posTMP[:, 2], 'g')
#ax3[1,1].set_xlabel('Y [AU]')
#ax3[1,1].set_ylabel('Z [AU]')

ax3 = plt.figure().add_subplot(projection='3d')
ax3.plot(pos_msI[:, 0], pos_msI[:, 1], pos_msI[:, 2], 'b', label='Multi Segment')
ax3.plot(r_CRTBP_I[:, 0], r_CRTBP_I[:, 1], r_CRTBP_I[:, 2], 'r', label='CRTBP')
ax3.plot(r_CRTBP2_I[:, 0], r_CRTBP2_I[:, 1], r_CRTBP2_I[:, 2], 'g-.', label='CRTBP w/ MS process')
ax3.scatter(pos_msI[0, 0], pos_msI[0, 1], pos_msI[0, 2])
ax3.scatter(r_CRTBP_I[0, 0], r_CRTBP_I[0, 1], r_CRTBP_I[0, 2])
ax3.set_title('FF vs CRTBP in I frame (Inertial EM)')
ax3.set_xlabel('X [AU]')
ax3.set_ylabel('Y [AU]')
ax3.set_zlabel('Z [AU]')
plt.legend()
#
#fig3, ax3 = plt.subplots(2,2)
#ax3[0,1].plot(pos_msI[:, 0], pos_msI[:, 1], 'b')
#ax3[0,1].plot(r_CRTBP_I[:, 0], r_CRTBP_I[:, 1], 'r')
#ax3[0,1].plot(posTMP[:, 0], posTMP[:, 1], 'g')
#ax3[0,1].set_xlabel('X [AU]')
#ax3[0,1].set_ylabel('Y [AU]')
#ax3[1,0].plot(pos_msI[:, 0], pos_msI[:, 2], 'b')
#ax3[1,0].plot(r_CRTBP_I[:, 0], r_CRTBP_I[:, 2], 'r')
#ax3[1,0].plot(posTMP[:, 0], posTMP[:, 2], 'g')
#ax3[1,0].set_xlabel('X [AU]')
#ax3[1,0].set_ylabel('Z [AU]')
#ax3[1,1].plot(pos_msI[:, 1], pos_msI[:, 2], 'b')
#ax3[1,1].plot(r_CRTBP_I[:, 1], r_CRTBP_I[:, 2], 'r')
#ax3[1,1].plot(posTMP[:, 1], posTMP[:, 2], 'g')
#ax3[1,1].set_xlabel('Y [AU]')
#ax3[1,1].set_ylabel('Z [AU]')

ax4 = plt.figure().add_subplot(projection='3d')
ax4.plot(posR[:, 0], posR[:, 1], posR[:, 2], 'b', label='Multi Segment')
ax4.plot(r_CRTBP_rot[:, 0], r_CRTBP_rot[:, 1], r_CRTBP_rot[:, 2], 'r',label='CRTBP')
#ax4.plot(r_CRTBP2_rot[:, 0], r_CRTBP2_rot[:, 1], r_CRTBP2_rot[:, 2], 'g', label='CRTBP w/ MS process')
ax4.scatter(posR[0, 0], posR[0, 1], posR[0, 2])
ax4.scatter(r_CRTBP_rot[0, 0], r_CRTBP_rot[0, 1], r_CRTBP_rot[0, 2])
ax4.set_title('FF vs CRTBP in R frame (Rotating)')
ax4.set_xlabel('X [AU]')
ax4.set_ylabel('Y [AU]')
ax4.set_zlabel('Z [AU]')
plt.legend()

#fig3, ax3 = plt.subplots(2,2)
#ax3[0,1].plot(posR[:, 0], posR[:, 1], 'b')
#ax3[0,1].plot(r_CRTBP_rot[:, 0], r_CRTBP_rot[:, 1], 'r')
#ax3[0,1].plot(posTMP[:, 0], posTMP[:, 1], 'g')
#ax3[0,1].set_xlabel('X [AU]')
#ax3[0,1].set_ylabel('Y [AU]')
#ax3[1,0].plot(posR[:, 0], posR[:, 2], 'b')
#ax3[1,0].plot(r_CRTBP_rot[:, 0], r_CRTBP_rot[:, 2], 'r')
#ax3[1,0].plot(posTMP[:, 0], posTMP[:, 2], 'g')
#ax3[1,0].set_xlabel('X [AU]')
#ax3[1,0].set_ylabel('Z [AU]')
#ax3[1,1].plot(posR[:, 1], posR[:, 2], 'b')
#ax3[1,1].plot(r_CRTBP_rot[:, 1], r_CRTBP_rot[:, 2], 'r')
#ax3[1,1].plot(posTMP[:, 1], posTMP[:, 2], 'g')
#ax3[1,1].set_xlabel('Y [AU]')
#ax3[1,1].set_ylabel('Z [AU]')

plt.show()
breakpoint()

# Preallocate space
r_PEM_r = np.zeros([len(timesFF), 3])
r_SunEM_r = np.zeros([len(timesFF), 3])
r_EarthEM_r = np.zeros([len(timesFF), 3])
r_MoonEM_r = np.zeros([len(timesFF), 3])

# sim time in mjd
timesFF_mjd = timesFF + t_start

# Obtain Moon, Earth, and Sun positions for FF
for ii in np.arange(len(timesFF)):
    time = timesFF_mjd[ii]

    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU').value
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU').value
    EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)
    r_EMO = EMO[0].get_xyz().to('AU').value

    # Convert from H frame to GCRS frame
    r_PG = frameConversion.icrs2gmec(posFF[ii]*u.AU, time)
    r_EMG = frameConversion.icrs2gmec(r_EMO*u.AU, time)
    r_SunG = frameConversion.icrs2gmec(r_SunO*u.AU, time)
    r_MoonG = frameConversion.icrs2gmec(r_MoonO*u.AU, time)

    # Change the origin to the EM barycenter, G frame
    r_PEM = r_PG - r_EMG
    r_SunEM = r_SunG - r_EMG
    r_EarthEM = -r_EMG
    r_MoonEM = r_MoonG - r_EMG

    # Convert from G frame to I frame
    r_PEM_r[ii, :] = C_G2I@r_PEM.to('AU')
    r_SunEM_r[ii, :] = C_G2I@r_SunEM.to('AU')
    r_EarthEM_r[ii, :] = C_G2I@r_EarthEM.to('AU')
    r_MoonEM_r[ii, :] = C_G2I@r_MoonEM.to('AU')


plt.show()
# breakpoint()

