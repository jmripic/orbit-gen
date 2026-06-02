
import numpy as np
import astropy.units as u
from astropy.time import Time
import datetime


def extractSTK(file_path):
    """Extracts position and time data from an exported GMAT file

    Args:
        file_path (str):
            Path to the file you want to extract position data from. File contains data of the form
            [x_position [km], y_position [km], z_position [km], time [UTCG]]

    Returns:
        pos (float n array):
            Position array in AU
        time (astropy Time array):
            Time vector in MJD

    """

    x_positions = []
    y_positions = []
    z_positions = []
    times = []

    # Read the file
    with open(file_path, 'r') as file:
        # Skip the header lines
        for line in file:
            if line.startswith("-"):
                break

        # Process each line of data
        for line in file:
            columns = line.split()
            if len(columns) < 4:
                continue

            # Extract x, y, z positions
            x_positions.append(float(columns[0]))
            y_positions.append(float(columns[1]))
            z_positions.append(float(columns[2]))

            # Extract and reformat time to 'YYYY-MM-DD HH:MM:SS.sss'
            day = columns[3]
            month = columns[4]
            year = columns[5]
            time_part = columns[6] if len(columns) > 6 else "00:00:00.000"
            # Convert to datetime object and reformat
            date_obj = datetime.datetime.strptime(f"{day} {month} {year} {time_part}", "%d %b %Y %H:%M:%S.%f")
            times.append(date_obj.strftime("%Y-%m-%d %H:%M:%S.%f"))

    # Convert to AU and convert lists to numpy arrays
    x_positions = (x_positions * u.km).to('AU')
    y_positions = (y_positions * u.km).to('AU')
    z_positions = (z_positions * u.km).to('AU')
    positions = np.array([x_positions, y_positions, z_positions]).T  # 3D array for position vectors
    # Convert times to MJD using Astropy
    time_array = Time(times, format='iso', scale='utc').mjd
    time_array = Time(time_array, format='mjd', scale='utc')

    return positions, time_array


def extract_pos(filepath):
    """Extracts position and time data from an exported GMAT file

    Args:
        filepath (str):
            Path to the file you want to extract position data from. File contains data of the form
            [x_position, y_position, z_position, time]

    Returns:
        pos (float n array):
            Position array in km
        time (astropy Time array):
            Time vector in MJD

    """

    data = []
    with open(filepath) as file:
        next(file)
        for line in file:
            row = line.split()
            row = [float(x) for x in row]
            data.append(row)

    x = list(map(lambda x: x[0], data)) * u.km
    y = list(map(lambda x: x[1], data)) * u.km
    z = list(map(lambda x: x[2], data)) * u.km
    pos = np.array([x, y, z]).T
    time = Time(list(map(lambda x: x[3], data)), format='mjd', scale='utc')

    return pos, time


def extract_posvel(filepath):
    """Extracts position and velocity data from an exported GMAT file

    Args:
        filepath (str):
            Path to the file you want to extract position data from. File contains data of the form
            [x_position, y_position, z_position, x_velocity, y_velocity, z_velocity, time]

    Returns:
        pos (float n array):
            Position array in km
        vel (float n array):
            Velocity array in km/s
        time (astropy Time array):
            Time vector in MJD

    """

    data = []
    with open(filepath) as file:
        next(file)
        for line in file:
            row = line.split()
            row = [float(x) for x in row]
            data.append(row)

    x = list(map(lambda x: x[0], data)) * u.km
    y = list(map(lambda x: x[1], data)) * u.km
    z = list(map(lambda x: x[2], data)) * u.km
    vx = list(map(lambda x: x[3], data)) * (u.km/u.s)
    vy = list(map(lambda x: x[4], data)) * (u.km/u.s)
    vz = list(map(lambda x: x[5], data)) * (u.km/u.s)

    pos = np.array([x, y, z]).T
    vel = np.array([vx, vy, vz]).T

    time = Time(list(map(lambda x: x[6], data)), format='mjd', scale='utc')

    return pos, vel, time
