
import os.path
import numpy as np
import pickle
import math
from scipy.integrate import tplquad
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import sys


# ~~~~RETRIEVE DATA FROM FILE~~~~

sys.path.insert(1, 'orbits')
path_str = "orbitFiles/DRO_11.241_days.p"  # Change to whichever orbit you'd like
path_f1 = os.path.normpath(os.path.expandvars(path_str))
f1 = open(path_f1, "rb")  # "rb" means "read binary"
orbit_data = pickle.load(f1)  # open the pickled file (and unpickle it)
f1.close()  # close the file

# ~~~~DEFINE CONSTANTS~~~~

barycenter_to_earth = -4641  # x coordinate [km]
barycenter_to_moon = 379764  # x coordinate [km]
earth_to_moon = 384405  # x coordinate [km]
earth_radius = 6378  # [km]
moon_radius = 1738  # [km]


# ~~~~A METHOD THAT MAKES ALL AXES EQUAL IN THE PLOT~~~~
def set_axes_equal(ax):
    # Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc.
    # Input ax: a matplotlib axis, e.g., as output from plt.gca().

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# ~~~~A CLASS WITH METHODS THAT EVALUATE THE ORBIT~~~~
class VisibilityMetric:

    def __init__(self, orbit):
        self.orbit = orbit  # Orbit is the UNPICKLED data

    def visible_altitude(self, y_fov, z_fov, altitude):
        # Calculates the percentage of the chosen altitude that is visible from the spacecraft's orbit.
        # Returns a vector (percent_visible) with that percentage at each orbital position.
        # y_fov is the field of view in the y direction, in degrees.
        # z_fov is the field of view in the z direction, in degrees.
        # altitude is the altitude from the moon that you want to observe, in kilometers.

        # ~~~~CALL THE COMPONENTS~~~~
        # state: position in km and velocity km/s stored as rows
        # t: times states are defined at in s
        # te: orbit period in s
        # x_lpoint: km from the Earth-Moon barycenter

        x_position = np.array(self.orbit["state"][:, 0])  # From barycenter
        # this calls the first element in the first array of 'state'
        y_position = np.array(self.orbit["state"][:, 1])
        z_position = np.array(self.orbit["state"][:, 2])
        x_velocity = np.array(self.orbit["state"][:, 3])
        y_velocity = np.array(self.orbit["state"][:, 4])
        z_velocity = np.array(self.orbit["state"][:, 5])
        L2_point = np.array(self.orbit["x_lpoint"])
        time = np.array(self.orbit["t"])

        alt_radius = altitude + moon_radius

        x_from_moon = [abs(x - barycenter_to_moon) for x in x_position]
        y_from_moon = y_position  # [km]
        z_from_moon = z_position  # [km]
        sc_to_moon = []  # Magnitude of the distance between the spacecraft and the moon

        i = 0
        for x in x_from_moon:
            sc_to_moon.append(np.sqrt(x_from_moon[i] ** 2 + y_from_moon[i] ** 2 + z_from_moon[i] ** 2))  # [km]
            i = i + 1
        sc_to_moon = np.array(sc_to_moon)

        # ~~~~VOLUME OF THE MOON~~~~
        # (Not really necessary, just for practice...)

        r1_moon, r2_moon = 0, moon_radius
        theta1_moon, theta2_moon = 0, 2*np.pi
        phi1_moon, phi2_moon = 0, np.pi

        def diff_volume1(phi, theta, r):
            return (r**2)*sin(phi)

        volume_moon = tplquad(diff_volume1, r1_moon, r2_moon, lambda r: theta1_moon, lambda r: theta2_moon,
                              lambda r, theta: phi1_moon, lambda r, theta: phi2_moon)[0]

        # ~~~~VOLUME OF THE DESIRED ALTITUDE (no moon included)~~~~

        r1_alt, r2_alt = 0, alt_radius
        theta1_alt, theta2_alt = 0, 2*np.pi
        phi1_alt, phi2_alt = 0, np.pi

        volume_alt = (tplquad(diff_volume1, r1_alt, r2_alt, lambda r: theta1_alt, lambda r: theta2_alt,
                              lambda r, theta: phi1_alt, lambda r, theta: phi2_alt)[0]) - volume_moon

        # ~~~~VOLUME OF THE WHAT IS VISIBLE FROM THE SPACECRAFT~~~~

        target_volume = []
        i = 0
        for x in sc_to_moon:
            if sc_to_moon[i] < alt_radius:  # If the spacecraft is within the target altitude
                x1, x2 = 0, abs(sc_to_moon[i] - moon_radius)
            else:  # If the spacecraft is outside the target altitude
                x1, x2 = abs(sc_to_moon[i] - alt_radius), abs(sc_to_moon[i] - moon_radius)
            y1, y2 = lambda x: -np.tan(math.radians(y_fov / 2)) * x, lambda x: np.tan(math.radians(y_fov / 2)) * x
            z1, z2 = lambda x, y: -np.tan(math.radians(z_fov / 2)) * x, lambda x, y: np.tan(math.radians(z_fov / 2)) * x
            target_volume.append(tplquad(lambda x, y, z: 1, x1, x2, y1, y2, z1, z2)[0])  # [km]
            i = i+1
        target_volume = np.array(target_volume)
        # The second argument in the result is an estimate of the error

        # ~~~~PERCENT VISIBLE~~~~
        # Percent of the desired altitude that is visible from the spacecraft

        percent_visible = (target_volume/volume_alt)*100
        plt.plot(time, percent_visible)
        plt.ylabel('Percentage of the altitude that is visible from the spacecraft')
        plt.xlabel('Time [s]')
        plt.show()


    def plot_fov(self, y_fov, z_fov, altitude, i):
        # Plots the moon, the target altitude, the orbit, and the field of view that the orbit sees.
        # i is the position of the orbit that you want to plot the field of view from.

        # ~~~~CALL THE COMPONENTS~~~~
        # state: position in km and velocity km/s stored as rows
        # t: times states are defined at in s
        # te: orbit period in s
        # x_lpoint: km from the Earth-Moon barycenter

        x_position = np.array(self.orbit["state"][:, 0])  # From barycenter
        # this calls the first element in the first array of 'state'
        y_position = np.array(self.orbit["state"][:, 1])
        z_position = np.array(self.orbit["state"][:, 2])
        x_velocity = np.array(self.orbit["state"][:, 3])
        y_velocity = np.array(self.orbit["state"][:, 4])
        z_velocity = np.array(self.orbit["state"][:, 5])
        L2_point = np.array(self.orbit["x_lpoint"])
        time = np.array(self.orbit["t"])

        alt_radius = altitude + moon_radius

        x_from_moon = [abs(x - barycenter_to_moon) for x in x_position]
        y_from_moon = y_position  # [km]
        z_from_moon = z_position  # [km]
        sc_to_moon = []  # Magnitude of the distance between the spacecraft and the moon

        k = 0
        for x in x_from_moon:
            sc_to_moon.append(np.sqrt(x_from_moon[k] ** 2 + y_from_moon[k] ** 2 + z_from_moon[k] ** 2))  # [km]
            k = k + 1
        sc_to_moon = np.array(sc_to_moon)

        # Set up the figure
        fig = plt.figure()
        ax3D = fig.add_subplot(projection="3d")  # Call a 3D plot
        ax3D.set_xlabel('X Position [km]')  # Label axes
        ax3D.set_ylabel('Y Position [km]')
        ax3D.set_zlabel('Z Position [km]')

        # Plot the orbit
        ax3D.scatter3D(x_position, y_position, z_position, s=1)  # Plot the orbit

        # Plot the surface of the moon
        phi = np.linspace(0, 2*np.pi, 500)  # Angle phi
        theta = np.linspace(0, np.pi, 500)  # Angle theta

        x_moon = barycenter_to_moon + (moon_radius * np.outer(np.cos(phi), np.sin(theta)))
        y_moon = moon_radius * np.outer(np.sin(phi), np.sin(theta))
        z_moon = moon_radius * np.outer(np.ones(np.size(phi)), np.cos(theta))

        ax3D.plot_surface(x_moon, y_moon, z_moon, color='lightgreen')

        # Plot the surface of the desired altitude
        x_alt = barycenter_to_moon + (alt_radius * np.outer(np.cos(phi), np.sin(theta)))
        y_alt = alt_radius * np.outer(np.sin(phi), np.sin(theta))
        z_alt = alt_radius * np.outer(np.ones(np.size(phi)), np.cos(theta))

        ax3D.plot_surface(x_alt, y_alt, z_alt, alpha=0.1)

        # Plot the field of view from the spacecraft
        # Currently for 1 orbital position at a time. Can be modified to plot the field of view for every
        # orbital position, if necessary.
        # NOTE: This section contains an issue with the rotation of the field of view. For any index except for the
        # first and last orbital positions, the field of view does not hit the moon. I suspect that it's because
        # the second rotation is around the body axes instead of the space axes, but I haven't been able to
        # figure out why this is happening.

        x_line = [x_position[i]-sc_to_moon[i], x_position[i]]  # "Horizontal" line from SC to moon
        y_line1 = []
        y_line2 = []
        z_line1 = []
        z_line2 = []

        j = 0  # Index of each point in the field of view vectors
        for x in x_line:  # For each element in the x position vector, calculate the y and z vectors

            # Non-rotated vectors, pointed towards the negative x direction
            y_line1.append((np.tan(math.radians(y_fov / 2)) * (x_line[j]-x_position[i])) + y_position[i])
            y_line2.append((-np.tan(math.radians(y_fov / 2)) * (x_line[j]-x_position[i])) + y_position[i])
            z_line1.append((np.tan(math.radians(z_fov / 2)) * (x_line[j]-x_position[i])) + z_position[i])
            z_line2.append((-np.tan(math.radians(z_fov / 2)) * (x_line[j]-x_position[i])) + z_position[i])

            j = j+1

        # Angles to rotate the vectors
        theta_fov = np.arctan(z_from_moon[i]/x_from_moon[i])  # Rotation about the y-axis [rad]
        phi_fov = np.arctan(y_from_moon[i]/x_from_moon[i])  # Rotation about the z-axis [rad]

        # Create the DCM
        DCM_y = np.array([[np.cos(theta_fov), 0, -np.sin(theta_fov)], [0, 1, 0], [np.sin(theta_fov), 0, np.cos(theta_fov)]])
        DCM_z = np.array([[np.cos(phi_fov), -np.sin(phi_fov), 0], [np.sin(phi_fov), np.cos(phi_fov), 0], [0, 0, 1]])
        DCM = np.dot(DCM_y, DCM_z)

        # Apply the rotation
        fov11 = []
        fov12 = []
        fov22 = []
        fov21 = []

        j = 0
        fov11 = (np.dot(DCM, np.array([[x_line[j]-x_position[i]], [y_line1[j]-y_position[i]], [z_line1[j]-z_position[i]]])) +
                 np.array([[x_position[i]], [y_position[i]], [z_position[i]]]))
        fov12 = (np.dot(DCM, np.array([[x_line[j]-x_position[i]], [y_line1[j]-y_position[i]], [z_line2[j]-z_position[i]]])) +
                 np.array([[x_position[i]], [y_position[i]], [z_position[i]]]))
        fov22 = (np.dot(DCM, np.array([[x_line[j]-x_position[i]], [y_line2[j]-y_position[i]], [z_line2[j]-z_position[i]]])) +
                 np.array([[x_position[i]], [y_position[i]], [z_position[i]]]))
        fov21 = (np.dot(DCM, np.array([[x_line[j]-x_position[i]], [y_line2[j]-y_position[i]], [z_line1[j]-z_position[i]]])) +
                 np.array([[x_position[i]], [y_position[i]], [z_position[i]]]))

        # Plot
        verts1 = [list(zip([x_position[i], fov11[0, 0], fov12[0, 0]], [y_position[i], fov11[1, 0], fov12[1, 0]], [z_position[i], fov11[2, 0], fov12[2, 0]]))]
        verts2 = [list(zip([x_position[i], fov21[0, 0], fov11[0, 0]], [y_position[i], fov21[1, 0], fov11[1, 0]], [z_position[i], fov21[2, 0], fov11[2, 0]]))]
        verts3 = [list(zip([x_position[i], fov22[0, 0], fov21[0, 0]], [y_position[i], fov22[1, 0], fov21[1, 0]], [z_position[i], fov22[2, 0], fov21[2, 0]]))]
        verts4 = [list(zip([x_position[i], fov12[0, 0], fov22[0, 0]], [y_position[i], fov12[1, 0], fov22[1, 0]], [z_position[i], fov12[2, 0], fov22[2, 0]]))]

        ax3D.add_collection3d(Poly3DCollection(verts1, linewidth=0.2, edgecolors='black', facecolors='purple', alpha=0.5))
        ax3D.add_collection3d(Poly3DCollection(verts2, linewidth=0.2, edgecolors='black', facecolors='purple', alpha=0.5))
        ax3D.add_collection3d(Poly3DCollection(verts3, linewidth=0.2, edgecolors='black', facecolors='purple', alpha=0.5))
        ax3D.add_collection3d(Poly3DCollection(verts4, linewidth=0.2, edgecolors='black', facecolors='purple', alpha=0.5))

        # Show the plot
        ax3D.set_box_aspect([1.0, 1.0, 1.0])
        set_axes_equal(ax3D)
        plt.show()


orbit = VisibilityMetric(orbit_data)  # Creates an object in the class

y_fov = 1
z_fov = 1
altitude = 560

# orbit.visible_altitude(y_fov, z_fov, altitude)

orbital_position_index = 50
orbit.plot_fov(y_fov, z_fov, altitude, orbital_position_index)
