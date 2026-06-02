import numpy as np
import astropy.units as u
import sys
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import datetime
sys.path.insert(1, 'tools')
import extractTools
import frameConversion
import plot_tools
import unitConversion
from scipy.spatial.transform import Rotation as R

# ~~~~~IMPORTING STK DATA~~~~~

file_path = "gmatSTKFiles/L2Orbit_Position_Data_Rot_200.txt"
pos_rot, times = extractTools.extractSTK(file_path)
breakpoint()

# Convert to I frame from R frame
t_equinox = Time(51544.5, format='mjd', scale='utc')
t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
t_start = Time(57727, format='mjd', scale='utc')

pos_convert = np.zeros([len(times), 3])
for ii in np.arange(len(times)):
    C_I2R = frameConversion.inert2rot(times[ii], times[0])
    C_R2I = C_I2R.T
    pos_convert[ii, :] = C_R2I @ pos_rot[ii, :]

title = 'STK Spacecraft'
body_names = ['Rot to inert convert [AU]']
fig_I, ax_I = plot_tools.plot_bodies(pos_convert, body_names=body_names, title=title)


# ~~~~~FUNCTIONS FOR EQUAL TIME INTERVALS~~~~~

# Resample positions to equal time intervals
def resample_positions(positions, times, new_times):
    resampled_positions = np.zeros((len(new_times), 3))
    for i in range(3):  # Interpolate for x, y, z separately
        resampled_positions[:, i] = np.interp(new_times.mjd, times.mjd, positions[:, i])
    return resampled_positions


# Calculate DCM given two position vectors
def calculate_dcm(pos1, pos2):
    # Normalizing position vectors
    pos1_unit = pos1 / np.linalg.norm(pos1)
    pos2_unit = pos2 / np.linalg.norm(pos2)

    # Define rotation axis and angle
    axis = np.cross(pos1_unit, pos2_unit)
    if np.linalg.norm(axis) < 1e-8:  # Check for nearly parallel vectors
        return np.eye(3)  # Return identity if vectors are parallel
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(pos1_unit, pos2_unit))

    # Create DCM using scipy Rotation
    dcm = R.from_rotvec(angle * axis).as_matrix()
    return dcm


# Compare DCMs over time within a single frame
def compare_dcms(positions, times, interval):
    # Create a new time array with equal intervals
    start_time = times[0]
    end_time = times[-1]
    delta_time = interval * u.s  # Interval in seconds

    new_times = Time(np.arange(start_time.mjd, end_time.mjd, delta_time.to(u.day).value), format='mjd')

    resampled_positions = resample_positions(positions, times, new_times)

    # Calculate a DCM between each position vector
    dcms = []
    for i in range(1, len(resampled_positions)):
        dcm = calculate_dcm(resampled_positions[i - 1], resampled_positions[i])
        dcms.append(dcm)

    # Compare DCMs (using Frobenius norm)
    differences = []
    for i in range(1, len(dcms)):
        diff = np.linalg.norm(dcms[i] - dcms[i - 1], ord='fro')  # Frobenius norm
        differences.append(diff)

    return differences, new_times[2:]


# # ~~~~~GET DATA~~~~~
#
# t_equinox = Time(51544.5, format='mjd', scale='utc')
# t_veq = t_equinox + 79.3125*u.d  # + 1*u.yr/4
# t_start = Time(57727, format='mjd', scale='utc')
#
# # Get GMAT data
# file_name = "gmatFiles/Moon_EMInert.txt"  # MJ2000Eq centered at EM barycenter
# moon_inert, times = gmatTools.extract_pos(file_name)  # Array in km, time in MJD
# moon_inert = np.array((moon_inert * u.km).to('AU'))  # Array in AU
#
# file_name = "gmatFiles/Moon_EMRot.txt"  # Rotating with the moon centered at EM barycenter
# moon_rot, _ = gmatTools.extract_pos(file_name)  # Array in km
# moon_rot = np.array((moon_rot * u.km).to('AU'))  # Array in AU
#
# file_name = "gmatFiles/SC_EMInert.txt"  # MJ2000Eq centered at EM barycenter
# sc_inert, _ = gmatTools.extract_pos(file_name)  # Array in km
# sc_inert = np.array((sc_inert * u.km).to('AU'))  # Array in AU
#
# file_name = "gmatFiles/SC_EMRot.txt"  # Rotating with the moon centered at EM barycenter
# sc_rot, _ = gmatTools.extract_pos(file_name)  # Array in km
# sc_rot = np.array((sc_rot * u.km).to('AU'))  # Array in AU
#
# # Get Astropy moon data (I frame, AU)
# C_I2G = frameConversion.inert2geo(t_start, t_veq)
# astro_moon_inert = np.zeros([len(times), 3])
# for ii in np.arange(len(times)):
#     _, _, astro_moon_inert[ii, :] = frameConversion.getSunEarthMoon(times[ii], C_I2G)


# ~~~~~CONVERT GMAT~~~~~

# # Convert to I frame from R frame
# sc_inertconvert = np.zeros([len(times), 3])
# moon_inertconvert = np.zeros([len(times), 3])
# C_I2G = frameConversion.inert2geo(times[0], t_veq)
# for ii in np.arange(len(times)):
#     C_I2R = frameConversion.inert2rot_GMAT(times[ii], times[0], C_I2G)
#     C_R2I = C_I2R.T
#     sc_inertconvert[ii, :] = C_R2I @ sc_rot[ii, :]
#     moon_inertconvert[ii, :] = C_R2I @ moon_rot[ii, :]


# # ~~~~~APPLY FUNCTIONS~~~~~
#
# # Compare DCM between each point in inertial frame EQUAL TIME
# interval = 3600  # Time interval in seconds for a new, even time array (1 hour)
# gmat_differences, plot_times = compare_dcms(moon_inert, times, interval)
# astro_differences, _ = compare_dcms(astro_moon_inert, times, interval)
#
# # Compare inertial and rotating frame
# dcms_norm = []
# for i in range(0, len(moon_inert)):
#     dcm = calculate_dcm(moon_inert[i], moon_rot[i])
#     dcm_norm = np.linalg.norm(dcm, ord='fro')
#     dcms_norm.append(dcm_norm)


# ~~~~~PLOT~~~~~

# # Compare converted data
# title = 'Spacecraft in the Inertial Frame'
# body_names = ['Inertial', 'Converted']
# fig_I, ax_I = plot_tools.plot_bodies(sc_inert, sc_inertconvert, body_names=body_names, title=title)
#
# title = 'Moon in the Inertial Frame'
# body_names = ['Inertial', 'Converted']
# fig_I, ax_I = plot_tools.plot_bodies(moon_inert, moon_inertconvert, body_names=body_names, title=title)


# # Compare DCM between each point in inertial frame FOR EQUAL TIME INTERVALS
# plt.figure(figsize=(10, 6))
# plt.plot(plot_times.value, gmat_differences, marker='.', linestyle='-', color='b', label="GMAT")
# plt.plot(plot_times.value, astro_differences, marker='.', linestyle='-', color='r', label="Astropy")
#
# plt.xlabel('Time [MJD]')
# plt.ylabel('Frobenius Norm of DCM Differences')
# plt.title('Differences in DCMs Over Time (equal time intervals)')
# plt.grid(True)
# plt.legend()


# # Compare inertial and rotating frame
# plt.figure(figsize=(10, 6))
# plt.plot(times.value, dcms_norm, marker='.', linestyle='-', color='b')
#
# plt.xlabel('Time [MJD]')
# plt.ylabel('DCM')
# plt.title('DCM Between Inertial and Rotating Frame Over Time in GMAT')
# plt.grid(True)

# plt.show()

# breakpoint()
