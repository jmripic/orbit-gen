import numpy as np
import sys
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import extractTools
import pdb

# ~~~~~PROPAGATE THE DYNAMICS~~~~~

# Initialize the kernel
coord.solar_system.solar_system_ephemeris.set('de440')

# Parameters
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')
days = 100
days_can = unitConversion.convertTime_to_canonical(days * u.d)
mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

# Initial condition in canonical units in rotating frame R [pos, vel]
IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]  # L2, 5.92773293-day period
# IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2]  # DRO, 0.9697497-day period

# Generate new ICs using the free variable and constraint method
X = [IC[0], IC[2], IC[4], IC[6]]
max_iter = 1000
error = 10
ctr = 0
eps = 4E-6
while error > eps and ctr < max_iter:
    Fx = orbitEOMProp.calcFx_R(X, mu_star)

    error = np.linalg.norm(Fx)
    dFx = orbitEOMProp.calcdFx_CRTBP(X, mu_star, m1, m2)

    X = X - dFx.T @ (np.linalg.inv(dFx @ dFx.T) @ Fx)

    ctr = ctr + 1

IC = np.array([X[0], 0, X[1], 0, X[2], 0, 2*X[3]])  # Canonical, rotating frame

# DCM for G frame and I frame
C_I2G = frameConversion.inert2geo(t_start, t_veq)
C_G2I = C_I2G.T

# Get position of the moon at the epoch in the inertial frame
sun_I, earth_I, moon_I = frameConversion.getSunEarthMoon(t_start, C_I2G)  # I frame [AU]
moon_I_can = unitConversion.convertPos_to_canonical(moon_I)

# Transform position ICs to the epoch moon
ideal_moon = [1-mu_star, 0, 0]
IC_x = (IC[0] - ideal_moon[0]) + moon_I_can[0]
IC_y = (IC[1] - ideal_moon[1]) + moon_I_can[1]
IC_z = (IC[2] - ideal_moon[2]) + moon_I_can[2]
IC[0:3] = [IC_x, IC_y, IC_z]  # Canonical, I frame

# Convert the velocity to I frame from R frame (position is the same in both)
vO = frameConversion.rot2inertV(np.array(IC[0:3]), np.array(IC[3:6]), 0)

# Rotate velocity vector to match the epoch moon (I frame)
theta = np.arccos((np.dot(moon_I_can, ideal_moon))/(np.linalg.norm(moon_I_can)*np.linalg.norm(ideal_moon)))
if theta > np.pi/2:
    theta = -theta
rot_matrix = frameConversion.rot(theta, 3)
IC[3:6] = rot_matrix @ vO  # Canonical, I frame

# Convert IC to dimensional, rotating frame (for STK)
C_I2R = frameConversion.inert2rot(t_start, t_start)
pos_canrot = C_I2R @ IC[0:3]  # Canonical, R frame
vel_canrot = frameConversion.inert2rotV(pos_canrot, IC[3:6], 0)  # Canonical, R frame
pos_dimrot = unitConversion.convertPos_to_dim(pos_canrot).to('km')  # R frame, dimensional
vel_dimrot = unitConversion.convertVel_to_dim(vel_canrot).to('km/s')  # R frame, dimensional
print('Dimensional [km] position IC in the rotating frame: ', pos_dimrot)
print('Dimensional [km/s] velocity IC in the rotating frame: ', vel_dimrot)

# Convert ICs to H frame (AU and AU/d) from I frame (canonical)
pos_H, vel_H = frameConversion.convertSC_I2H(IC[0:3], IC[3:6], t_start, C_I2G)

# Define the initial state array
state0 = np.append(np.append(pos_H.value, vel_H.value), days)

# Propagate the dynamics (states in AU or AU/day, times in days starting from 0)
states, times = orbitEOMProp.statePropFF(state0, t_start)  # State is in the H frame
pos = states[:, 0:3]
vel = states[:, 3:6]

# Convert to canonical
pos_can = unitConversion.convertPos_to_canonical(pos * u.AU)
vel_can = unitConversion.convertVel_to_canonical(vel * u.AU/u.d)

# Simulation time in mjd
times_mjd = times + t_start  # Days from mission start time

# Preallocate space
pos_SC = np.zeros([len(times_mjd), 3])
vel_SC = np.zeros([len(times_mjd), 3])
pos_Sun = np.zeros([len(times_mjd), 3])
pos_Earth = np.zeros([len(times_mjd), 3])
pos_Moon = np.zeros([len(times_mjd), 3])
pos_Sun_H = np.zeros([len(times_mjd), 3])
pos_Earth_H = np.zeros([len(times_mjd), 3])
pos_Moon_H = np.zeros([len(times_mjd), 3])

# Obtain celestial body positions in the I frame [AU] and convert state to I frame
for ii in np.arange(len(times_mjd)):
    pos_SC[ii, :], vel_SC[ii, :] = frameConversion.convertSC_H2I(pos_can[ii, :], vel_can[ii, :], times_mjd[ii], C_I2G)
    pos_Sun[ii, :], pos_Earth[ii, :], pos_Moon[ii, :] = frameConversion.getSunEarthMoon(times_mjd[ii], C_I2G)
    pos_Sun_H[ii, :] = get_body_barycentric_posvel('Sun', times_mjd[ii])[0].get_xyz().to('AU').value
    pos_Earth_H[ii, :] = get_body_barycentric_posvel('Earth', times_mjd[ii])[0].get_xyz().to('AU').value
    pos_Moon_H[ii, :] = get_body_barycentric_posvel('Moon', times_mjd[ii])[0].get_xyz().to('AU').value


# ~~~~~OBTAIN STK DATA~~~~

# Obtain FF rotating data from STK
file_path = "gmatSTKFiles/L2Orbit_Full_Force_State_earthpoint100.txt"
stk_posrot, stk_times = extractTools.extractSTK(file_path)

# Convert to I frame from R frame
stk_posinert = np.zeros([len(stk_times), 3])
for ii in np.arange(len(stk_times)):
    C_I2R = frameConversion.inert2rot(stk_times[ii], stk_times[0])
    C_R2I = C_I2R.T
    stk_posinert[ii, :] = C_R2I @ stk_posrot[ii, :]


# ~~~~~PLOT~~~~

title = 'Full Force Model in the I Frame'
body_names = ['Propagated FF', 'Earth', 'Moon', 'STK Orbit']
fig_I, ax_I = plot_tools.plot_bodies(pos_SC, pos_Earth, pos_Moon, stk_posinert, body_names=body_names, title=title)

 title = 'Full Force Model in the H Frame'
 body_names = ['Spacecraft', 'Moon', 'Sun']
 fig_H, ax_H = plot_tools.plot_bodies(pos, pos_Moon_H, pos_Sun_H, body_names=body_names, title=title)


# ~~~~~ANIMATION~~~~~


def interpolate_positions(stk_pos, stk_times, target_times):
    # Create interpolation functions for each position component (x, y, z)
    interp_func_x = interp1d(stk_times.value, stk_pos[:, 0], kind='linear', fill_value="extrapolate")
    interp_func_y = interp1d(stk_times.value, stk_pos[:, 1], kind='linear', fill_value="extrapolate")
    interp_func_z = interp1d(stk_times.value, stk_pos[:, 2], kind='linear', fill_value="extrapolate")

    # Interpolate stk_posrot to match target_times
    interp_x = interp_func_x(target_times.value)
    interp_y = interp_func_y(target_times.value)
    interp_z = interp_func_z(target_times.value)

    # Combine interpolated components into a new position array
    interpolated_posrot = np.vstack((interp_x, interp_y, interp_z)).T

    return interpolated_posrot


interp_stk_posinert = interpolate_positions(stk_posinert, stk_times, times_mjd)

desired_duration = 5  # seconds
title = 'Full Force Model in the Inertial (I) Frame'
body_names = ['Propagated FF', 'Earth', 'Moon', 'STK Orbit']
animate_func_I, ani_object_I = plot_tools.create_animation(times, days, desired_duration,
                                                       [pos_SC, pos_Earth, pos_Moon, interp_stk_posinert],
                                                       body_names=body_names, title=title)

# title = 'Full Force Model in the H Frame'
# body_names = ['Spacecraft', 'Earth', 'Moon', 'Sun']
# animate_func_H, ani_object_H = plot_tools.create_animation(times, days, desired_duration,
#                                                            [pos, pos_Earth_H, pos_Moon_H, pos_Sun_H],
#                                                            body_names=body_names, title=title)


# # ~~~~~SAVE~~~~~
#fig_I.savefig('plotFigures/FF STK earth point mass 100 days.png')
fig_H.savefig('plotFigures/FF DRO H frame.png')

writergif = animation.PillowWriter(fps=30)
#ani_object_I.save('plotFigures/FF STK earth point mass 100 days.gif', writer=writergif)
ani_object_H.save('plotFigures/FF DRO H frame.gif', writer=writergif)
