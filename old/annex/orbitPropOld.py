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
import gmatTools
import plot_tools

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
t_veq = t_equinox + 79.3125*u.d
#t_start = Time(52027, format='mjd', scale='utc')
t_start = Time(57727, format='mjd', scale='utc')
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

C_I2G = frameConversion.inert2geo(t_start, t_veq)
C_G2I = C_I2G.T

# Initial condition in non-dimensional units in rotating frame R [pos, vel]
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
    
IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])  # Canonical, rotating frame

# Convert ICs to dimensional, rotating frame (for GMAT)
pos_dim = unitConversion.convertPos_to_dim(IC[0:3]).to('km')
breakpoint()
vel_dim = unitConversion.convertVel_to_dim(IC[3:6]).to('km/s')
print('Dimensional [km] position IC in the rotating frame: ', pos_dim)
print('Dimensional [km/s] velocity IC in the rotating frame: ', vel_dim)
    
# Convert the velocity to I frame from R frame
vI = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Define the free variable array
freeVar_CRTBP = np.array([IC[0], IC[2], vI[1], 1*IC[-1]])

# Propagate the dynamics in the CRTBP model for 1 orbit
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


# Convert position from I frame (canonical) to H frame [AU]
pos_H, vel_H = frameConversion.convertSC_I2H(posCRTBP[0], velCRTBP[0], t_start, C_I2G)

# Define the initial state array (for ~200 day orbit)
state0 = np.append(np.append(pos_H.value, vel_H.value), 33*times_dim[-1].value)   # Change to Tp_dim.value for one orbit

# Propagate the dynamics in the full force model (H frame) [AU, AU/d, days from 0]
statesFF, timesFF = orbitEOMProp.statePropFF(state0, t_start) #,times_dim)
posFF = statesFF[:, 0:3]
velFF = statesFF[:, 3:6]

# Preallocate space
r_PEM_i = np.zeros([len(timesFF), 3])
r_SunEM_i = np.zeros([len(timesFF), 3])
r_EarthEM_i = np.zeros([len(timesFF), 3])
r_MoonEM_i = np.zeros([len(timesFF), 3])
r_PEM_g = np.zeros([len(timesFF), 3])
r_SunEM_g = np.zeros([len(timesFF), 3])
r_EarthEM_g = np.zeros([len(timesFF), 3])
r_MoonEM_g = np.zeros([len(timesFF), 3])
r_EMO_h = np.zeros([len(timesFF), 3])
r_SunO_h = np.zeros([len(timesFF), 3])
r_EarthO_h = np.zeros([len(timesFF), 3])
r_MoonO_h = np.zeros([len(timesFF), 3])

# sim time in mjd
timesFF_mjd = Time(timesFF + t_start.value, format='mjd', scale='utc')

# Obtain Moon, Earth, and Sun positions for FF
for ii in np.arange(len(timesFF_mjd)):
    time = timesFF_mjd[ii]

    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU')
    r_EarthO = get_body_barycentric_posvel('Earth', time)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU')
    r_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)[0].get_xyz().to('AU')

    r_SunO_h[ii, :] = r_SunO.value
    r_EarthO_h[ii, :] = r_EarthO.value
    r_MoonO_h[ii, :] = r_MoonO.value
    r_EMO_h[ii, :] = r_EMO.value

    # Convert from H frame to GMEc frame
    r_PG = frameConversion.icrs2gmec(posFF[ii]*u.AU, time)
    r_EMG2 = frameConversion.icrs2gmec(r_EMO, time)
    r_SunG = frameConversion.icrs2gmec(r_SunO, time)
    r_MoonG = frameConversion.icrs2gmec(r_MoonO, time)
    
    r_PEM_g[ii, :] = (r_PG - r_EMG2).to('AU').value
#    r_PEM_g[ii, :] = (r_PG).to('AU').value
    r_SunEM = (r_SunG - r_EMG2).to('AU').value
    r_EarthEM = -r_EMG2.to('AU').value
    r_MoonEM = (r_MoonG - r_EMG2).to('AU').value
        
    r_PEM_i[ii, :] = C_G2I@r_PEM_g[ii, :]
    r_EarthEM_i[ii, :] = C_G2I@r_EarthEM
    r_SunEM_i[ii, :] = C_G2I@r_SunEM
    r_MoonEM_i[ii, :] = C_G2I@r_MoonEM
    
#    breakpoint()


# Obtain CRTBP data from GMAT
file_name = "gmatFiles/FF_rot.txt"
gmat_km, gmat_time = gmatTools.extract_pos(file_name)
gmat_posrot = np.array((gmat_km * u.km).to('AU'))

# Convert to I frame from R frame
gmat_posinert = np.zeros([len(gmat_time), 3])
gmat_posgme = np.zeros([len(gmat_time), 3])
gmat_posicrs = np.zeros([len(gmat_time), 3])
for ii in np.arange(len(gmat_time)):
    C_I2R = frameConversion.inert2rot(gmat_time[ii], gmat_time[0])
    C_R2I = C_I2R.T
    gmat_posinert[ii, :] = C_R2I @ gmat_posrot[ii, :]
    gmat_posgme[ii, :] = C_I2G @ gmat_posinert[ii, :]
    tmp1 = get_body_barycentric_posvel('Earth-Moon-Barycenter', gmat_time[ii])[0].get_xyz().to('AU')
    tmp1 = frameConversion.icrs2gmec(tmp1, gmat_time[ii]).to('AU')
    tmp2 = gmat_posgme[ii, :]*u.AU + tmp1
    gmat_posicrs[ii, :] = frameConversion.gmec2icrs(tmp2, gmat_time[ii]).to('AU').value
    

# Plot
ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(posFF[:, 0], posFF[:, 1], posFF[:, 2], 'b', label='Full Force')
ax1.plot(r_PO_CRTBP[:,0], r_PO_CRTBP[:,1], r_PO_CRTBP[:,2], 'r-.', label='CRTBP')
ax1.plot(gmat_posicrs[:, 0], gmat_posicrs[:, 1], gmat_posicrs[:, 2], color='g', label='GMAT Orbit')
ax1.scatter(posFF[0, 0], posFF[0, 1], posFF[0, 2], c='b', marker='*', label='Full Force Start')
ax1.scatter(posFF[-1, 0], posFF[-1, 1], posFF[-1, 2], c='b', marker='D', label='Full Force End')
ax1.scatter(r_PO_CRTBP[0, 0], r_PO_CRTBP[0, 1], r_PO_CRTBP[0, 2], c='r', marker='*', label='CRTBP Start')
ax1.scatter(r_PO_CRTBP[-1, 0], r_PO_CRTBP[-1, 1], r_PO_CRTBP[-1, 2], c='r', marker='D', label='CRTBP End')
ax1.set_title('FF vs CRTBP in H frame (ICRS)')
ax1.set_xlabel('X [AU]')
ax1.set_ylabel('Y [AU]')
ax1.set_zlabel('Z [AU]')
plt.legend()

#fig, axs = plt.subplots(3)
#fig.suptitle('ICRS differences')
#axs[0].plot(timesFF, posFF[:,0]-r_PO_CRTBP[:,0])
#axs[0].set_ylabel('x [AU]')
#axs[1].plot(timesFF, posFF[:,1]-r_PO_CRTBP[:,1])
#axs[1].set_ylabel('y [AU]')
#axs[2].plot(timesFF, posFF[:,2]-r_PO_CRTBP[:,2])
#axs[2].set_ylabel('z [AU]')
#axs[2].set_xlabel('time [d]')
#
#fig, axs = plt.subplots(3)
#fig.suptitle('ICRS differences')
#axs[0].plot(timesFF[:635], posFF[:635,0]-r_PO_CRTBP[:635,0])
#axs[0].set_ylabel('x [AU]')
#axs[1].plot(timesFF[:635], posFF[:635,1]-r_PO_CRTBP[:635,1])
#axs[1].set_ylabel('y [AU]')
#axs[2].plot(timesFF[:635], posFF[:635,2]-r_PO_CRTBP[:635,2])
#axs[2].set_ylabel('z [AU]')
#axs[2].set_xlabel('time [d]')


ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(r_PEM_i[:, 0], r_PEM_i[:, 1], r_PEM_i[:, 2], 'b', label='Full Force')
ax2.plot(gmat_posinert[:, 0], gmat_posinert[:, 1], gmat_posinert[:, 2], color='g', label='GMAT Orbit')
ax2.plot(r_CRTBP_I[:, 0], r_CRTBP_I[:, 1], r_CRTBP_I[:, 2], 'r-.', label='CRTBP')
ax2.plot(r_EarthEM_i[:, 0], r_EarthEM_i[:, 1], r_EarthEM_i[:, 2], 'g', label='Earth')
ax2.plot(r_MoonEM_i[:, 0], r_MoonEM_i[:, 1], r_MoonEM_i[:, 2], 'k', label='Moon')
#ax2.scatter(r_PEM_i[0, 0], r_PEM_i[0, 1], r_PEM_i[0, 2], c='b', marker='*', label='Full Force Start')
#ax2.scatter(r_PEM_i[-1, 0], r_PEM_i[-1, 1], r_PEM_i[-1, 2], c='b', marker='D', label='Full Force End')
#ax2.scatter(r_CRTBP_I[0, 0], r_CRTBP_I[0, 1], r_CRTBP_I[0, 2], c='r', marker='*', label='CRTBP Start')
#ax2.scatter(r_CRTBP_I[-1, 0], r_CRTBP_I[-1, 1], r_CRTBP_I[-1, 2], c='r', marker='D', label='CRTBP End')
ax2.set_title('FF vs CRTBP in I frame (Inertial EM)')
ax2.set_xlabel('X [AU]')
ax2.set_ylabel('Y [AU]')
ax2.set_zlabel('Z [AU]')
plt.legend()

desired_duration = 3  # seconds
title = 'Full Force Model in the Inertial (I) Frame'
body_names = ['Propagated FF', 'Earth', 'Moon', 'Sun']
animate_func, ani_object = plot_tools.create_animation(timesFF, 33*IC[-1], desired_duration,
                                                       [r_PEM_i, r_EarthEM_i, r_MoonEM_i, r_SunEM_i], body_names=body_names,
                                                       title=title)

desired_duration = 3  # seconds
title = 'Full Force Model in the Inertial (I) Frame, no Sun'
body_names = ['Propagated FF', 'Earth', 'Moon']
animate_func2, ani_object2 = plot_tools.create_animation(timesFF, 33*IC[-1], desired_duration,
                                                       [r_PEM_i, r_EarthEM_i, r_MoonEM_i], body_names=body_names,
                                                       title=title)
# Save
writergif = animation.PillowWriter(fps=30)
ani_object2.save('FF L2 orbitProp no sun.gif', writer=writergif)

#fig, axs = plt.subplots(3)
#fig.suptitle('I frame differences')
#axs[0].plot(timesFF, r_PEM_i[:,0]-r_CRTBP_I[:,0])
#axs[0].set_ylabel('x [AU]')
#axs[1].plot(timesFF, r_PEM_i[:,1]-r_CRTBP_I[:,1])
#axs[1].set_ylabel('y [AU]')
#axs[2].plot(timesFF, r_PEM_i[:,2]-r_CRTBP_I[:,2])
#axs[2].set_ylabel('z [AU]')
#axs[2].set_xlabel('time [d]')
#
#fig, axs = plt.subplots(3)
#fig.suptitle('I frame differences')
#axs[0].plot(timesFF[:635], r_PEM_i[:635,0]-r_CRTBP_I[:635,0])
#axs[0].set_ylabel('x [AU]')
#axs[1].plot(timesFF[:635], r_PEM_i[:635,1]-r_CRTBP_I[:635,1])
#axs[1].set_ylabel('y [AU]')
#axs[2].plot(timesFF[:635], r_PEM_i[:635,2]-r_CRTBP_I[:635,2])
#axs[2].set_ylabel('z [AU]')
#axs[2].set_xlabel('time [d]')

#ax3 = plt.figure().add_subplot(projection='3d')
#ax3.plot(r_PEM_i[:, 0], r_PEM_i[:, 1], r_PEM_i[:, 2], 'b', label='Full Force')
#ax3.plot(r_CRTBP_I[:,0], r_CRTBP_I[:,1], r_CRTBP_I[:,2], 'r-.', label='CRTBP')
#ax3.plot(r_EarthEM_i[:,0], r_EarthEM_i[:,1], r_EarthEM_i[:,2], 'g', label='Earth')
#ax3.plot(r_MoonEM_i[:,0], r_MoonEM_i[:,1], r_MoonEM_i[:,2], 'k', label='Moon')
#ax3.plot(r_SunEM_i[:,0], r_SunEM_i[:,1], r_SunEM_i[:,2], 'y', label='Sun')
#ax3.set_title('FF vs CRTBP in I frame (Inertial EM)')
#ax3.set_xlabel('X [AU]')
#ax3.set_ylabel('Y [AU]')
#ax3.set_zlabel('Z [AU]')
#plt.legend()

ax4 = plt.figure().add_subplot(projection='3d')
ax4.plot(r_PEM_g[:, 0], r_PEM_g[:, 1], r_PEM_g[:, 2], 'b', label='Full Force')
ax4.plot(gmat_posgme[:, 0], gmat_posgme[:, 1], gmat_posgme[:, 2], color='g', label='GMAT Orbit')
ax4.plot(r_CRTBP_G[:,0], r_CRTBP_G[:,1], r_CRTBP_G[:,2], 'r-.', label='CRTBP')
ax4.plot(r_EarthEM_g[:,0], r_EarthEM_g[:,1], r_EarthEM_g[:,2], 'g', label='Earth')
ax4.plot(r_MoonEM_g[:,0], r_MoonEM_g[:,1], r_MoonEM_g[:,2], 'k', label='Moon')
ax4.set_title('FF vs CRTBP in G frame (GME centered at EM)')
ax4.set_xlabel('X [AU]')
ax4.set_ylabel('Y [AU]')
ax4.set_zlabel('Z [AU]')
plt.legend()

#fig, axs = plt.subplots(3)
#fig.suptitle('G frame differences')
#axs[0].plot(timesFF, r_PEM_g[:,0]-r_CRTBP_G[:,0])
#axs[0].set_ylabel('x [AU]')
#axs[1].plot(timesFF, r_PEM_g[:,1]-r_CRTBP_G[:,1])
#axs[1].set_ylabel('y [AU]')
#axs[2].plot(timesFF, r_PEM_g[:,2]-r_CRTBP_G[:,2])
#axs[2].set_ylabel('z [AU]')
#axs[2].set_xlabel('time [d]')
#
#fig, axs = plt.subplots(3)
#fig.suptitle('G frame differences')
#axs[0].plot(timesFF[:635], r_PEM_g[:635,0]-r_CRTBP_G[:635,0])
#axs[0].set_ylabel('x [AU]')
#axs[1].plot(timesFF[:635], r_PEM_g[:635,1]-r_CRTBP_G[:635,1])
#axs[1].set_ylabel('y [AU]')
#axs[2].plot(timesFF[:635], r_PEM_g[:635,2]-r_CRTBP_G[:635,2])
#axs[2].set_ylabel('z [AU]')
#axs[2].set_xlabel('time [d]')

#ax5 = plt.figure().add_subplot(projection='3d')
#ax5.plot(r_PEM_g[:, 0], r_PEM_g[:, 1], r_PEM_g[:, 2], 'b', label='Full Force')
#ax5.plot(r_CRTBP_G[:,0], r_CRTBP_G[:,1], r_CRTBP_G[:,2], 'r-.', label='CRTBP')
#ax5.plot(r_EarthEM_g[:,0], r_EarthEM_g[:,1], r_EarthEM_g[:,2], 'g', label='Earth')
#ax5.plot(r_MoonEM_g[:,0], r_MoonEM_g[:,1], r_MoonEM_g[:,2], 'k', label='Moon')
#ax5.plot(r_SunEM_g[:,0], r_SunEM_g[:,1], r_SunEM_g[:,2], 'y', label='Sun')
#ax5.set_title('FF vs CRTBP in G frame (GCRS centered at EM)')
#ax5.set_xlabel('X [AU]')
#ax5.set_ylabel('Y [AU]')
#ax5.set_zlabel('Z [AU]')
#plt.legend()
#
#ax6 = plt.figure().add_subplot(projection='3d')
#ax6.plot(r_SunEM_g[:,0], r_SunEM_g[:,1], r_SunEM_g[:,2], 'b', label='Sun G')
#ax6.plot(r_SunEM_i[:,0], r_SunEM_i[:,1], r_SunEM_i[:,2], 'g', label='Sun I')
#ax6.set_xlabel('X [AU]')
#ax6.set_ylabel('Y [AU]')
#ax6.set_zlabel('Z [AU]')
#plt.legend()
#
#ax7 = plt.figure().add_subplot(projection='3d')
#ax7.plot(r_EarthEM_g[:,0], r_EarthEM_g[:,1], r_EarthEM_g[:,2], 'b', label='Earth G')
#ax7.plot(r_EarthEM_i[:,0], r_EarthEM_i[:,1], r_EarthEM_i[:,2], 'g', label='Earth I')
#ax7.set_xlabel('X [AU]')
#ax7.set_ylabel('Y [AU]')
#ax7.set_zlabel('Z [AU]')
#plt.legend()

#ax6 = plt.figure().add_subplot(projection='3d')
#ax6.plot(r_EarthO_h[:,0], r_EarthO_h[:,1], r_EarthO_h[:,2], 'g', label='Earth')
#ax6.plot(r_EMO_h[:,0], r_EMO_h[:,1], r_EMO_h[:,2], 'r', label='Earth-Moon Barycenter')
#ax6.plot(r_MoonO_h[:,0], r_MoonO_h[:,1], r_MoonO_h[:,2], 'k', label='Moon')
#ax6.set_title('H frame (ICRS)')
#ax6.set_xlabel('X [AU]')
#ax6.set_ylabel('Y [AU]')
#ax6.set_zlabel('Z [AU]')
#plt.legend()

#ax7 = plt.figure().add_subplot(projection='3d')
#ax7.plot(r_EarthO_h[:,0], r_EarthO_h[:,1], r_EarthO_h[:,2], 'g', label='Earth')
#ax7.plot(r_EMO_h[:,0], r_EMO_h[:,1], r_EMO_h[:,2], 'r', label='Earth-Moon Barycenter')
#ax7.plot(r_MoonO_h[:,0], r_MoonO_h[:,1], r_MoonO_h[:,2], 'k', label='Moon')
#ax7.plot(r_SunO_h[:,0], r_SunO_h[:,1], r_SunO_h[:,2], 'y', label='Sun')
#ax7.set_title('H frame (ICRS)')
#ax7.set_xlabel('X [AU]')
#ax7.set_ylabel('Y [AU]')
#ax7.set_zlabel('Z [AU]')
#plt.legend()

plt.show()
breakpoint()

