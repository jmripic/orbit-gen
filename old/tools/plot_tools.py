import numpy as np
from matplotlib import pyplot as plt
import constants as c
import frameConversion
import unitConversion
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import astropy.units as u
from scipy.interpolate import interp1d
from matplotlib import animation

## temporary, for anna
#import starlift.orbits.tools.unitConversion as unitConversion
#import starlift.orbits.tools.frameConversion as frameConversion
#import starlift.orbits.tools.constants as c


def Orbit3D(solvec, time, args={}):
    """Plot the orbit in three dimensions. Default origin is the EM barycenter in the EM synodic reference frame. The dimensioned argument is also supplied to properly organize the display of the Earth, Moon, and axis scaling. 
    args={'Frame': 'Synodic', 'dimensioned':False}"""

    _args = {'Frame': 'Synodic', 'dimensioned':True}
    for key in args.keys():
        _args[ key ] = args[ key ]

    x_vals = np.array(solvec[:,0])
    y_vals = np.array(solvec[:,1])
    z_vals = np.array(solvec[:,2])

    ax = plt.axes(projection='3d')
    traj = ax.scatter(x_vals,y_vals,z_vals, c=time, cmap = 'plasma', s=.5)
    ax.scatter(0,0,0, c='m', marker='*')
    
    n = np.linspace(0,2*np.pi,100)
    v = np.linspace(0, np.pi, 100)

    if _args['dimensioned'] == False:
        re = c.earthR / c.lstar
        rm = c.moonR / c.lstar
        eoffset = -c.mustar
        moffset = 1-c.mustar
    else:
        re = c.earthR
        rm = c.moonR
        eoffset = -c.mustar*c.moonSMA
        moffset = (1-c.mustar)*c.moonSMA

    xe = re * np.outer(np.cos(n), np.sin(v)) + eoffset
    ye = re * np.outer(np.sin(n), np.sin(v))
    ze = re * np.outer(np.ones(np.size(n)), np.cos(v))

    xm = rm * np.outer(np.cos(n), np.sin(v)) + moffset
    ym = rm * np.outer(np.sin(n), np.sin(v))
    zm = rm * np.outer(np.ones(np.size(n)), np.cos(v))

    ax.plot_surface(xe,ye,ze)
    ax.plot_surface(xm,ym,zm)
    plt.title('Orbit in the Earth-Moon Rotating Frame')

    plt.axis('equal')
    ax.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(traj)
    plt.show()


def PlotManifold(solvec, time, mu, ax, title,eigval, eigvec):
    # _args = {'Frame': 'Synodic'}
    x_vals = np.array(solvec[0,:])
    y_vals = np.array(solvec[1,:])
    z_vals = np.array(solvec[2,:])
    
    # ax = plt.axes(projection='3d')
    traj = ax.scatter(x_vals,y_vals,z_vals, c=time, cmap = 'plasma',s=.5)
    ax.scatter(0,0,0, c='m', marker='*')

    n = np.linspace(0,2*np.pi,100)
    v = np.linspace(0, np.pi, 100)

    re = c.earthD / c.lstar
    rm = c.moonD / c.lstar

    xe = re * np.outer(np.cos(n), np.sin(v)) - mu
    ye = re * np.outer(np.sin(n), np.sin(v)) + 0
    ze = re * np.outer(np.ones(np.size(n)), np.cos(v)) + 0

    xm = rm * np.outer(np.cos(n), np.sin(v)) + (1-mu)
    ym = rm * np.outer(np.sin(n), np.sin(v))
    zm = rm * np.outer(np.ones(np.size(n)), np.cos(v))

    ax.plot_surface(xe,ye,ze)
    ax.plot_surface(xm,ym,zm)
    plt.suptitle(title)

    ax.set_title(("Eigenvalue: ", eigval ))
    plt.axis('equal')
    # ax.text2D(0.05, 0.95, (r'Eigenvalue: ', eigval, r'\nEigvec: ', eigvec), transform=ax.transAxes)
    ax.legend()
    plt.xlabel('X\n')
    plt.ylabel('Y\n')
    # plt.colorbar(traj)


def plotConvertBodies(timesFF, posFF, t_mjd, frame, C_G2I):
    # ** Add documentation



    # preallocate space
    r_PEM_r = np.zeros([len(timesFF), 3])
    r_SunEM_r = np.zeros([len(timesFF), 3])
    r_EarthEM_r = np.zeros([len(timesFF), 3])
    r_MoonEM_r = np.zeros([len(timesFF), 3])

    # sim time in mjd
    timesFF_mjd = timesFF + t_mjd
    for ii in np.arange(len(timesFF)):
        time = timesFF_mjd[ii]

        if frame == 0:
            # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
            r_SunO = get_body_barycentric_posvel('Sun', time)[0].get_xyz().to('AU')
            r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU')
            r_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time).get_xyz().to('AU')
        
            # convert from H frame to GCRS frame
            r_PG = frameConversion.icrs2gmec(posFF[ii]*u.AU, time)
            r_EMG = frameConversion.icrs2gmec(r_EMO, time)
            r_SunG = frameConversion.icrs2gmec(r_SunO, time)
            r_MoonG = frameConversion.icrs2gmec(r_MoonO, time)
            
            # change the origin to the EM barycenter, G frame
            r_PEM = r_PG - r_EMG
            r_SunEM = r_SunG - r_EMG
            r_EarthEM = -r_EMG
            r_MoonEM = r_MoonG - r_EMG
            
            r_PEM_r[ii, :] = r_PEM
            r_SunEM_r[ii, :] = r_SunEM
            r_EarthEM_r[ii, :] = r_EarthEM
            r_MoonEM_r[ii, :] = r_MoonEM
            
        elif frame >= 1:
            # positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
            r_MoonO = get_body_barycentric_posvel('Moon', time)[0].get_xyz().to('AU')
            r_EMO = get_body_barycentric_posvel('Earth-Moon-Barycenter', time)[0].get_xyz().to('AU')
            
            # convert from H frame to GCRS frame
            r_PG = frameConversion.icrs2gmec(posFF[ii]*u.AU, time)
            r_EMG = frameConversion.icrs2gmec(r_EMO, time)
            r_MoonG = frameConversion.icrs2gmec(r_MoonO, time)
            
            # change the origin to the EM barycenter, G frame
            r_PEM = r_PG - r_EMG
            r_EarthEM = -r_EMG
            r_MoonEM = r_MoonG - r_EMG
            
            # convert from G frame to I frame
            r_PEM = C_G2I@r_PEM.to('AU')
            r_EarthEM = C_G2I@r_EarthEM.to('AU')
            r_MoonEM = C_G2I@r_MoonEM.to('AU')
            
            if frame == 2:
                C_I2R = frameConversion.inert2rot(time, t_mjd)
                r_PEM = C_I2R@r_PEM.to('AU')
                r_EarthEM = C_I2R@r_EarthEM.to('AU')
                r_MoonEM = C_I2R@r_MoonEM.to('AU')

            r_PEM_r[ii, :] = r_PEM.value
            r_EarthEM_r[ii, :] = r_EarthEM.value
            r_MoonEM_r[ii, :] = r_MoonEM.value

    return r_PEM_r, r_SunEM_r, r_EarthEM_r, r_MoonEM_r

        
def plotBodiesFF(timesFF, posFF, t_mjd, frame):
    # ** Add documentation

    r_PEM, r_SunEM, r_EarthEM, r_MoonEM = plotConvertBodies(timesFF, posFF, t_mjd, frame)
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(r_EarthEM[:, 0], r_EarthEM[:, 1], r_EarthEM[:, 2], 'g', label='Earth')
    ax.plot(r_MoonEM[:, 0], r_MoonEM[:, 1], r_MoonEM[:, 2], 'r', label='Moon')
    ax.plot(r_PEM[:, 0], r_PEM[:, 1], r_PEM[:, 2], 'b', label='Full Force')
    if frame == 0:
        ax.plot(r_SunEM[:, 0], r_SunEM[:, 1], r_SunEM[:, 2], 'y', label='Sun')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    
    # ** Add lines to resize figure and automatically save png and svg
    return


def plotCompare_rot(timesFF, posFF, t_mjd, frame, timesCRTBP, posCRTBP, C_G2I):
    # ** Add documentation
    
    r_PEM, _, r_EarthEM, r_MoonEM = plotConvertBodies(timesFF, posFF, t_mjd, frame, C_G2I)

    posCRTBP = (unitConversion.convertPos_to_dim(posCRTBP).to('AU')).value
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(posCRTBP[:, 0], posCRTBP[:, 1], posCRTBP[:, 2], 'k', label='CRTBP')
    ax.plot(r_PEM[:, 0], r_PEM[:, 1],r_PEM[:, 2], 'b', label='Full Force')
    ax.plot(r_EarthEM[:, 0], r_EarthEM[:, 1], r_EarthEM[:, 2], 'g', label='Earth')
    ax.plot(r_MoonEM[:, 0], r_MoonEM[:, 1], r_MoonEM[:, 2], 'r', label='Moon')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    breakpoint()
    # ** Add lines to resize figure and automatically save png and svg
    return


def plotCompare_inert(timesCRTBP, posCRTBP, t_mjd, timesFF, posFF, mu_star):
    # Fix this. Determine which inertial frame this should be
    # ** Add documentation
    
    times = timesCRTBP + t_mjd
    pos_dim = unitConversion.convertPos_to_dim(posCRTBP).to('AU')
    
    C_I2G = frameConversion.inert2geo(t_mjd)
    
    pos_H = np.zeros([len(times), 3])
    for ii in np.arange(len(timesCRTBP)):
        currentTime = times[ii]
        C_I2R = frameConversion.inert2rot(currentTime, t_mjd)
        pos_I = C_I2R @ pos_dim[ii]
        
        pos_G = C_I2G @ pos_I
        
        state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_mjd)
        posEMB = state_EMB[0].get_xyz().to('AU')
        velEMB = state_EMB[1].get_xyz().to('AU/day')
        posE = get_body_barycentric_posvel('Earth', t_mjd)[0].get_xyz().to('AU')
        posE_EMB = posE - posEMB

        pos_GCRS = pos_G - posE_EMB
        
        pos_H[ii, :] = (frameConversion.gmec2icrs(pos_GCRS, t_mjd)).to('AU')

    r_EarthO = get_body_barycentric_posvel('Earth', times)[0].get_xyz().to('AU')
    r_MoonO = get_body_barycentric_posvel('Moon', times)[0].get_xyz().to('AU')
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pos_H[:, 0], pos_H[:, 1], pos_H[:, 2], 'k', label='CRTBP')
    ax.plot(posFF[:, 0], posFF[:, 1],posFF[:, 2], 'b', label='Full Force')
#    ax.plot(r_EarthO[:, 0], r_EarthO[:, 1], r_EarthO[:, 2], 'g', label='Earth')
#    ax.plot(r_MoonO[:, 0], r_MoonO[:, 1], r_MoonO[:, 2], 'r', label='Moon')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    breakpoint()
    # ** Add lines to resize figure and automatically save png and svg
    return


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc.

    Example usage:
        ax.set_box_aspect([1.0, 1.0, 1.0])
        set_axes_equal(ax)

    Args:
        ax
            A matplotlib axis, e.g., as output from plt.gca().
    """

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


def calculate_plot_limits(*positions):
    """
    Calculates the plot limit based on the position data of multiple bodies to ensure that the plot includes the
    entire range of the data. A single limit is returned to be used on all axes to create a square plot.

    Example usage:
        limit = calculate_plot_limits(pos_SC, pos_Earth, pos_Moon, pos_Sun)
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])

    Parameters:
        positions (list of numpy arrays)
            X, Y, and Z position data for a list of bodies [AU]

    Returns:
        limit (float)
            The limit value for the plot to encompass the entire range of the data.
    """

    # Initialize min and max values for each axis
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

    # Find min and max values across all positions
    for pos in positions:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        min_x = min(min_x, np.min(x))
        min_y = min(min_y, np.min(y))
        min_z = min(min_z, np.min(z))
        max_x = max(max_x, np.max(x))
        max_y = max(max_y, np.max(y))
        max_z = max(max_z, np.max(z))

    # Calculate the limit as the maximum distance from the origin to any data point
    limit_x = max(abs(min_x), abs(max_x))
    limit_y = max(abs(min_y), abs(max_y))
    limit_z = max(abs(min_z), abs(max_z))

    limit = max(limit_x, limit_y, limit_z)

    return limit


def plot_bodies(*positions, body_names=None, title='Celestial Bodies Plot'):
    """
    Plots an indeterminate number of bodies in a static 3D plot. Returns fig and ax to be used to save the plot, as
    in the following example:
        fig.savefig('FF L2.png')

    Args::
        positions (list of numpy arrays)
            X, Y, and Z position of various celestial bodies or spacecraft in AU
        body_names (list of strings)
            Names for the bodies. If None, default names 'Body 1', 'Body 2', etc. are used.
        title (string)
            A title for the plot. Default is 'Celestial Bodies Plot'.

    Returns:
        fig
            The matplotlib figure object.
        ax
            The matplotlib 3D axis object.
    """

    # Set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # If no body names are given
    if body_names is None:
        body_names = [f'Body {i + 1}' for i in range(len(positions))]

    # Plot each body's data
    for i, pos in enumerate(positions):
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=body_names[i])

    # Set axis limits based on calculated limit
    limit = calculate_plot_limits(*positions)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()
    plt.title(title)

    # Show the plot
    plt.show()

    return fig, ax


def create_animation(times, days, desired_duration, positions, body_names=None, title='Celestial Bodies Animation'):
    """
    This function generates an animation of multiple celestial bodies over a given animation duration.

    Example usage:
        title = 'Full Force Model in the Inertial (I) Frame'
        body_names = ['Spacecraft', 'Earth', 'Moon', 'Sun']
        animate_func, ani_object = create_animation(times, days, desired_duration,
                                                       [pos_SC, pos_Earth, pos_Moon, pos_Sun], body_names=body_names,
                                                       title=title)

    Args:
        times (float n array)
            A normalized (starting from zero) time vector in canonical units
        days (float)
            The duration of the orbit simulation, in days
        desired_duration (float)
            The desired duration of the animation, in seconds
        positions (list of numpy arrays)
            X, Y, and Z position data in AU of multiple bodies. The number of bodies given is variable (for example,
            the Earth and the Sun can be given, or just the Earth)
        body_names (list of strings, optional)
            A list of the names of the bodies being plotted. This is so the legend on the plot is accurate. It's
            important that the order of the names matches the order of the position data, as seen in the example above.
        title (string)
            A title for the plot. Default is 'Celestial Bodies Animation'.

    Returns:
        animate (function)
            A function used to update the animation frames
        ani (FuncAnimation object)
            A FuncAnimation object that controls the creation and playback of the animation. Useful if one wants to
            save the animation. For example:
                writergif = animation.PillowWriter(fps=30)
                ani.save('FF L2.gif', writer=writergif)
    """

    # Compute a constant time interval between frames for the animation
    interval = unitConversion.convertTime_to_canonical(days * u.d) / 1000

    # Generate new evenly spaced times
    new_times = np.arange(times[0], times[-1], interval)

    # Interpolate positions for all bodies
    interpolated_positions = []
    for pos in positions:
        interpolated_pos = [
            interp1d(times, pos[:, 0], kind='linear')(new_times),
            interp1d(times, pos[:, 1], kind='linear')(new_times),
            interp1d(times, pos[:, 2], kind='linear')(new_times)
        ]
        interpolated_positions.append(np.array(interpolated_pos))

    # Calculate skip factor such that the animation lasts for the desired duration
    total_frames = len(new_times)
    frame_rate = 30  # Frames per second
    skip_factor = max(1, total_frames // (desired_duration * frame_rate))

    # Set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # If no body names are given
    if body_names is None:
        body_names = [f'Body {i + 1}' for i in range(len(positions))]

    # Initialize lines for each body
    lines = []
    for i, interp_pos in enumerate(interpolated_positions):
        line, = ax.plot(interp_pos[0, 0:1], interp_pos[1, 0:1], interp_pos[2, 0:1], label=body_names[i])
        lines.append(line)

    # Set axis limits
    limit = calculate_plot_limits(*positions)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    plt.legend()

    # Define the animate function
    def animate(i):
        idx = i * skip_factor
        if idx >= len(new_times):
            return

        for j, line in enumerate(lines):
            interp_pos = interpolated_positions[j]
            line.set_data(interp_pos[0, :idx], interp_pos[1, :idx])
            line.set_3d_properties(interp_pos[2, :idx])

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=total_frames // skip_factor, interval=1, repeat=True)
    plt.title(title)
    plt.show()

    # # Debugging frame intervals
    # print('Interval: ', interval)
    # for ii in range(1, len(new_times)):
    #     print(new_times[ii] - new_times[ii-1])

    return animate, ani
