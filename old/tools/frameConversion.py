import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import GCRS, ICRS, GeocentricMeanEcliptic, GeocentricTrueEcliptic
from astropy.coordinates.solar_system import get_body_barycentric_posvel
import sys
sys.path.insert(1, 'tools')
import unitConversion

# From JPL Horizons
# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10^22 kg
# m_earth = 5.97219x10^24 kg


def rot(th, axis):
    """Finds the rotation matrix of angle th about the axis value

    Args:
        th (float):
            Rotation angle in radians
        axis (int):
            Integer value denoting rotation axis (1,2, or 3)

    Returns:
        ~numpy.ndarray(float):
            Rotation matrix

    """

    if axis == 1:
        rot_th = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(th), np.sin(th)],
                [0.0, -np.sin(th), np.cos(th)],
            ]
        )
    elif axis == 2:
        rot_th = np.array(
            [
                [np.cos(th), 0.0, -np.sin(th)],
                [0.0, 1.0, 0.0],
                [np.sin(th), 0.0, np.cos(th)],
            ]
        )
    elif axis == 3:
        rot_th = np.array(
            [
                [np.cos(th), np.sin(th), 0.0],
                [-np.sin(th), np.cos(th), 0.0],
                [0.0, 0.0, 1.0]
            ]
        )

    return rot_th

    
def equinoxAngle(r_LAAN, r_veq, t_LAAN, t_veq):
    """Finds the angle between the GMECL equinox and the moon's ascending node

    Args:
        r_LAAN (astropy Quantity array):
            Longitude of the ascending node vector in Geocentric Mean Ecliptic frame
            in arbitrary distance units
        r_veq (astropy Quantity array):
            Vernal equinnox vector in Geocentric Mean Ecliptic frame in arbitrary
            distance units
        t_LAAN (astropy Time array):
            Longitude of the ascending node time in MJD
        t_veq (astropy Time array):
            Vernal equinox time in MJD

    Returns:
        theta (float):
            Angle between the two vectors in rad

    """
    
    n_veq = r_veq/np.linalg.norm(r_veq)
    n_LAAN = r_LAAN/np.linalg.norm(r_LAAN)

    dt = (t_LAAN - t_veq).value
    t_mod = np.mod(dt, 27.321582)
    if t_mod < 27.321582/2:
        sign = 1
    elif t_mod > 27.321582/2 and t_mod < 27.321582:
        sign = -1

    r_sin = np.linalg.norm(np.cross(n_LAAN, n_veq))
    r_cos = np.dot(n_LAAN, n_veq)
    theta = np.arctan2(sign*r_sin, r_cos)

    return theta


def rotAngle(currentTime, startTime):
    """Finds the angle of rotation between two vectors in any Earth-Moon-Barycenter centered frame

    Args:
        currentTime (astropy Time array):
            Current mission time in MJD
        startTime (astropy Time array):
            Mission start time in MJD

    Returns:
        theta (float):
            Angle between the two vectors in rad

    """
    
    
    r_M_ct = get_body_barycentric_posvel('Moon', currentTime)[0].get_xyz()
    r_M_st = get_body_barycentric_posvel('Moon', startTime)[0].get_xyz()

    r_Moon_ct = -icrs2gmec(r_M_ct, currentTime)
    r_Moon_st = -icrs2gmec(r_M_st, startTime)

    norm_ct = np.linalg.norm(r_Moon_ct)
    norm_st = np.linalg.norm(r_Moon_st)

    n_vec = np.cross(r_Moon_ct, r_Moon_st.T)
    
    dt = (currentTime - startTime).value
    t_mod = np.mod(dt, 27.321582)
    if t_mod < 27.321582/2:
        sign = 1
    elif t_mod > 27.321582/2 and t_mod < 27.321582:
        sign = -1
    r_sin = (np.linalg.norm(n_vec)/(norm_ct*norm_st))
    r_cos = (np.dot(r_Moon_ct/norm_ct, r_Moon_st.T/norm_st))
    theta = np.arctan2(sign*r_sin, r_cos)

    return theta


def rotMatAxisAng(n_vec, theta):
    """Computes a rotation matrix given an axis of rotation and an angle of rotation

    Args:
        n_hat (float n array)
            A unit vector specifying the axis of rotation (3D)
        theta (float)
            Angle of rotation in radians

    Returns:
        R (float n array)
            A 3x3 rotation matrix

    """
    n_hat = n_vec/np.linalg.norm(n_vec)
    r_skew = np.array([[0, -n_hat[2], n_hat[1]],
                       [n_hat[2], 0, -n_hat[0]],
                       [-n_hat[1], n_hat[0], 0]])

    R = np.identity(3) + r_skew * np.sin(theta) + r_skew @ r_skew * (1 - np.cos(theta))

    return R
    

def inert2geo(startTime, t_veq):
    """Computes the DCM to go from the inertial Earth-Moon CRTBP frame
    (I frame) to the GeocentricMeanEcliptic frame centered at the Earth-Moon barycenter
    (G frame)

    Args:
        startTime (astropy Time array):
            Mission start time in MJD
        t_veq (astropy Time array):
            Vernal equinox time for 2000 in MJD

    Returns:
        C_I2G (float n array):
            3x3 Array for the directional cosine matrix

    """
    # coarse search for LAAN vector
    tarray = startTime + np.arange(28)*u.d
    r_moon = get_body_barycentric_posvel('Moon', tarray)[0].get_xyz()

    ctr = 0
    r_m = np.zeros([len(tarray), 3])
    for ii in tarray:
        r_m[ctr, :] = icrs2gmec(r_moon[:, ctr], ii).to('AU').value
        ctr = ctr + 1
    
    ZZ = r_m[:, 2]
    signZ = np.sign(ZZ)
    diffZ = np.diff(signZ)
    indZ = np.argwhere(2 == diffZ)[0][0]
    
    # fine search for LAAN vector
    t1 = tarray[indZ]
    t2 = tarray[indZ + 1]
    dt = (t2 - t1)/2
    t3 = t1 + dt

    r_moon1 = (get_body_barycentric_posvel('Moon', t1)[0].get_xyz()).to('AU')
    r_moon2 = (get_body_barycentric_posvel('Moon', t2)[0].get_xyz()).to('AU')
    r_moon3 = (get_body_barycentric_posvel('Moon', t3)[0].get_xyz()).to('AU')
    r_m1 = icrs2gmec(r_moon1, t1)
    r_m2 = icrs2gmec(r_moon2, t2)
    r_m3 = icrs2gmec(r_moon3, t3)

    error = r_m3[2]

    while np.abs(error.value) > 1E-8:
        sign1 = np.sign(r_m1[2])
        sign2 = np.sign(r_m2[2])
        sign3 = np.sign(r_m3[2])

        if sign3 == sign1:
            t1 = t3
            r_m1 = r_m3
            
            dt = (t2 - t1)/2
            t3 = t3 + dt

            if sign1 == sign2:
                # if here something went wrong
                print('if')
                breakpoint()
            
        elif sign3 == sign2:
            t2 = t3
            r_m2 = r_m3
            
            dt = (t2 - t1)/2
            t3 = t1 + dt

            if sign1 == sign2:
                # if here something went wrong
                print('elif')
                breakpoint()
            
        else:
            # if here something went wrong
            breakpoint()
            
        r_m = (get_body_barycentric_posvel('Moon', t3)[0].get_xyz()).to('AU')
        r_m3 = icrs2gmec(r_m, t3)
            
        error = r_m3[2]
        
    t_LAAN = t3
    moon_LAAN = get_body_barycentric_posvel('Moon', t_LAAN)[0].get_xyz()
    
    r_LAAN = icrs2gmec(moon_LAAN, t_LAAN)
    
    t_ss = t_veq + (1*u.yr).to('d')/4
    
    b1_h = get_body_barycentric_posvel('Sun', t_veq)[0].get_xyz()
    b2_h = get_body_barycentric_posvel('Sun', t_ss)[0].get_xyz()
    
    b1_g = icrs2gmec(b1_h, t_veq)
    b2_g = icrs2gmec(b2_h, t_ss)
    b3_g = np.cross(b1_g, b2_g).value

    theta_LAAN = equinoxAngle(r_LAAN, b1_g, t_LAAN, t_veq).value
    
    C_LAAN = rotMatAxisAng(b3_g, theta_LAAN)

    # find INC DCM
    tarray_r = startTime + np.arange(28)/1*u.d
    r_moons_r = get_body_barycentric_posvel('Moon', tarray_r)[0].get_xyz()

    r_m_g = np.zeros([len(tarray_r), 3])
    ctr = 0
    r_m_r = np.zeros([len(tarray_r), 3])

    for ii in tarray_r:
        r_m_g[ctr,:] = icrs2gmec(r_moons_r[:, ctr], ii).to('AU').value
        r_m_r[ctr,:] = C_LAAN @ icrs2gmec(r_moons_r[:, ctr], ii).to('AU').value
        ctr = ctr + 1

    n_INC = b1_g/np.linalg.norm(b1_g)
    
    theta_INC = -np.deg2rad(5.145)
    C_INC = rotMatAxisAng(n_INC, theta_INC)
    
    r_m_c = np.zeros([len(tarray_r), 3])
    ctr = 0
    for ii in tarray_r:
        r_m_c[ctr,:] = C_INC @ r_m_r[ctr,:]
        ctr = ctr + 1

    # find AOP DCM
    # rough search
    r_norm_r = np.linalg.norm(r_m_r, axis=1)
    r_min_r = min(r_norm_r)
    
    r_ind_r = np.argwhere(r_min_r == r_norm_r)[0][0]

    # fine search
    t_AOP_r = tarray_r[r_ind_r-1]
    tarray_f = t_AOP_r + 0.5*u.d + np.arange(1600)/800*u.d
    
    r_moons_f = get_body_barycentric_posvel('Moon', tarray_f)[0].get_xyz()

    ctr = 0
    r_m_f = np.zeros([len(tarray_f), 3])

    for ii in tarray_f:
        r_m_f[ctr,:] = C_INC @ C_LAAN @ icrs2gmec(r_moons_f[:, ctr], ii).to('AU').value
        ctr = ctr + 1
        
    r_norm_f = np.linalg.norm(r_m_f, axis=1)
    r_min_f = min(r_norm_f)
    
    r_ind_f = np.argwhere(r_min_f == r_norm_f)[0][0]
    t_AOP = tarray_f[r_ind_f]
    
    theta_AOP = rotAngle(t_AOP, t_LAAN).value
    
    n_AOP = np.array([0, 0, 1])
    C_AOP = rotMatAxisAng(n_AOP, theta_AOP)

    C_G2P = C_AOP @ C_INC @ C_LAAN

    ctr = 0
    r_m_e = np.zeros([len(tarray_r), 3])

    for ii in tarray_r:
        r_m_e[ctr,:] = C_AOP @ r_m_c[ctr,:]
        ctr = ctr + 1
        
    C_P2I = peri2inert(r_m_e[0,:])
    
    C_G2I = C_P2I @ C_G2P
    C_I2G = C_G2I.T
    
    return C_I2G
    
def peri2inert(pos):
    """Computes the DCM to go from the Earth-Moon perifocal frame
    (P frame) to the inertial Earth-Moon CRTBP frame centered at the Earth-Moon
    barycenter (I frame)

    Args:
        pos (astropy Quantity array):
            Position vector in P (perifocal) frame in arbitrary distance units

    Returns:
        C_P2I (float n array):
            3x3 Array for the directional cosine matrix

    """

    i1 = np.array([1, 0, 0])
    p1 = np.array([pos[0], pos[1], 0])
    p1 = p1/np.linalg.norm(p1)
    
    r_sin = np.linalg.norm(np.cross(i1, p1))
    r_cos = np.dot(i1, p1)
    theta = np.arctan2(r_sin, r_cos)
    C_P2I = rot(theta, 3)

    return C_P2I
    
    
def inert2rot(currentTime, startTime):
    """Compute the directional cosine matrix to go from the Earth-Moon CR3BP
    perifocal frame (I) to the Earth-Moon CR3BP rotating frame (R)
    
    Args:
        currentTime (astropy Time array):
            Current mission time in MJD
        startTime (astropy Time array):
            Mission start time in MJD

    Returns:
        C_I2R (float n array):
            3x3 Array for the directional cosine matrix
    """

    dt = currentTime.value - startTime.value
    theta = unitConversion.convertTime_to_canonical(dt*u.d)
    
    C_I2R = rot(theta, 3)
    
    return C_I2R


def icrs2rot(pos, currentTime, startTime, C_G2I):
    """Convert position vector in ICRS coordinate frame to rotating coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        startTime (astropy Time array):
            Mission start time in MJD
        C_G2I (float n array):
            3x3 Array for the directional cosine matrix to go from the G frame to the I frame

    Returns:
        r_rot (astropy Quantity array):
            Position vector in rotating frame in km
    """
    
    state_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', startTime)
    r_EMG_icrs = state_EM[0].get_xyz().to('AU')

    # pos = pos * u.AU
    r_PE_gmec = icrs2gmec(pos, currentTime)
    r_rot = r_PE_gmec
    r_EME_gmec = icrs2gmec(r_EMG_icrs, currentTime)
    r_PEM = r_PE_gmec - r_EME_gmec

    C_I2R = inert2rot(currentTime, startTime)
    
    r_rot = C_G2I@C_I2R@r_PEM

    return r_rot


def icrs2gmec(pos, currentTime, vel=None):
    """Convert position and velocity vectors in ICRS coordinate frame to Geocentric Mean Ecliptic coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in ICRS (solar system barycentric) frame in arbitrary distance units

    Returns:
        pos_gmec (astropy Quantity array):
            Position vector in Geocentric Mean Ecliptic frame in km
        vel_gmec (astropy Quantity array, optional):
            Velocity vector in Geocentric Mean Ecliptic frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_icrs = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='icrs', obstime=currentTime)
        state_gmec = state_icrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime))
        pos_gmec = state_gmec.cartesian.get_xyz()
        vel_gmec = state_gmec.velocity.get_d_xyz()
        return pos_gmec, vel_gmec
    else:
        pos = pos.to('km')
        pos_icrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='icrs', obstime=currentTime)
        pos_gmec = pos_icrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime)).cartesian.get_xyz()
        return pos_gmec


def icrs2gcrs(pos, currentTime, vel=None):
    """Convert position and velocity vectors in ICRS coordinate frame to GCRS frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in ICRS (heliocentric) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in ICRS (solar system barycentric) frame in arbitrary distance units

    Returns:
        pos_gmec (astropy Quantity array):
            Position vector in Geocentric Mean Ecliptic frame in km
        vel_gmec (astropy Quantity array, optional):
            Velocity vector in Geocentric Mean Ecliptic frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_icrs = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='icrs', obstime=currentTime)
        state_gcrs = state_icrs.transform_to(GCRS(obstime=currentTime))
        pos_gcrs = state_gcrs.cartesian.get_xyz()
        vel_gcrs = state_gcrs.velocity.get_d_xyz()
        return pos_gcrs, vel_gcrs
    else:
        pos = pos.to('km')
        pos_icrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='icrs', obstime=currentTime)
        pos_gcrs = pos_icrs.transform_to(GCRS(obstime=currentTime)).cartesian.get_xyz()
        return pos_gcrs
        

def gmec2gcrs(pos, currentTime, vel=None):
    """Convert position and velocity vectors in geocentric mean ecliptic coordinate frame to GCRS coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in GMEC (geocentric mean ecliptic) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in GMEC (geocentric mean ecliptic) frame in arbitrary distance units

    Returns:
        pos_gmec (astropy Quantity array):
            Position vector in GCRS frame in km
        vel_gmec (astropy Quantity array, optional):
            Velocity vector in GCRS frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_gmec = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='geocentricmeanecliptic', obstime=currentTime)
        state_gcrs = state_gmec.transform_to(GCRS(obstime=currentTime))
        pos_gcrs = state_gcrs.cartesian.get_xyz()
        vel_gcrs = state_gcrs.velocity.get_d_xyz()
        return pos_gcrs, vel_gcrs
    else:
        pos = pos.to('km')
        pos_gmec = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='geocentricmeanecliptic', obstime=currentTime)
        pos_gcrs = pos_gmec.transform_to(GCRS(obstime=currentTime)).cartesian.get_xyz()
        return pos_gcrs


def gcrs2gmec(pos, currentTime, vel=None):
    """Convert position and velocity vectors in GCRS coordinate frame to geocentric mean ecliptic coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in GCRS (geocentric mean ecliptic) frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in GCRS (geocentric mean ecliptic) frame in arbitrary distance units

    Returns:
        pos_gmec (astropy Quantity array):
            Position vector in geocentric mean ecliptic frame in km
        vel_gmec (astropy Quantity array, optional):
            Velocity vector in geocentric mean ecliptic frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_gcrs = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='gcrs', obstime=currentTime)
        state_gmec = state_gcrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime))
        pos_gmec = state_gmec.cartesian.get_xyz()
        vel_gmec = state_gmec.velocity.get_d_xyz()
        return pos_gmec, vel_gmec
    else:
        pos = pos.to('km')
        pos_gcrs = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='gcrs', obstime=currentTime)
        pos_gmec = pos_gcrs.transform_to(GeocentricMeanEcliptic(obstime=currentTime)).cartesian.get_xyz()
        return pos_gmec

def gmec2icrs(pos, currentTime, vel=None):
    """Convert position and velocity vectors in Geocentric Mean Ecliptic coordinate frame to ICRS coordinate frame
    
    Args:
        pos (astropy Quantity array):
            Position vector in Geocentric Mean Ecliptic frame in arbitrary distance units
        currentTime (astropy Time array):
            Current mission time in MJD
        vel (astropy Quantity array, optional):
            Velocity vector in Geocentric Mean Ecliptic frame in arbitrary distance units

    Returns:
        pos_icrs (astropy Quantity array):
            Position vector in ICRS frame in km
        vel_icrs (astropy Quantity array, optional):
            Velocity vector in ICRS frame in km/s
    """

    if vel is not None:
        pos = pos.to('km')
        vel = vel.to('km/s')
        state_gmec = coord.SkyCoord(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2],
                                    representation_type='cartesian', frame='geocentricmeanecliptic',
                                    obstime=currentTime)
        state_icrs = state_gmec.transform_to(ICRS())
        pos_icrs = state_icrs.cartesian.get_xyz()
        vel_icrs = state_icrs.velocity.get_d_xyz()
        return pos_icrs, vel_icrs
    else:
        pos = pos.to('km')
        pos_gmec = coord.SkyCoord(x=pos[0].value, y=pos[1].value, z=pos[2].value, unit='km',
                                  representation_type='cartesian', frame='geocentricmeanecliptic', obstime=currentTime)
        pos_icrs = pos_gmec.transform_to(ICRS()).cartesian.get_xyz()
        return pos_icrs


def rot2inertV(rR, vR, t_norm):
    """Convert velocity from rotating frame to inertial frame

    Args:
      rR (float nx3 array):
        Rotating frame position vectors in canonical units
      vR (float nx3 array):
        Rotating frame velocity vectors in canonical units
      t_norm (float):
        Normalized time units for current epoch

    Returns:
      float nx3 array:
        Inertial frame velocity vectors in canonical units
    """

    e3 = np.array([0, 0, 1])
    if rR.shape[0] == 3 and len(rR.shape) == 1:
        vI = vR + np.cross(e3, rR)
    else:
        vI = np.zeros([len(t_norm), 3])
        for t in range(len(t_norm)):
            vI[t, :] = vR[t, :] + np.cross(e3, rR[t, :])

    return vI

def inert2rotV(rR, vI, t_norm):
    """Convert velocity from inertial frame to rotating frame

    Args:
      rR (float nx3 array):
        Rotating frame position vectors in canonical units
      vI (float nx3 array):
        Inertial frame velocity vectors in canonical units
      t_norm (float):
        Normalized time units for current epoch

    Returns:
      float nx3 array:
        Rotating frame velocity vectors in canonical units
    """

    e3 = np.array([0, 0, 1])
    if rR.shape[0] == 3 and len(rR.shape) == 1:
        vR = vI - np.cross(e3, rR)
    else:
        vR = np.zeros([len(t_norm), 3])
        for t in range(len(t_norm)):
            vR[t, :] = vI[t, :] - np.cross(e3, rR[t, :])

    return vR
    
def inert2rotV2(rR, vI, t_norm, omega):
    """Convert velocity from inertial frame to rotating frame

    Args:
      rR (float nx3 array):
        Rotating frame position vectors in arbitrary units
      vI (float nx3 array):
        Inertial frame velocity vectors in arbitrary units
      t_norm (float):
        Arbitrary time units for current epoch
      omega (float):
        Rotation rate in arbitrary

    Returns:
      float nx3 array:
        Rotating frame velocity vectors in canonical units
    """

    e3 = np.array([0, 0, 1])*omega
    if rR.shape[0] == 3 and len(rR.shape) == 1:
        vR = vI - np.cross(e3, rR)
    else:
        vR = np.zeros([len(t_norm), 3])
        for t in range(len(t_norm)):
            vR[t, :] = vI[t, :] - np.cross(e3, rR[t, :])

    return vR


def convertSC_R2H(pos_R, vel_R, t_mjd, t_veq, Tp_can=None):
    """Converts initial conditions from the R frame to the H frame

    Args:
        pos_R (astropy Quantity array):
            Array of distance in canonical units
        vel_R (astropy Quantity array):
            Array of velocities in canonical units
        t_mjd (astropy Time array):
            Mission start time in MJD
        mu_star (float):
            Non-dimensional mass parameter
        Tp_can (float n array, optional):
            Optional array of times in canonical units

    Returns:
        tuple:
        pos_H (float n array):
            Array of distance in AU
        vel_H (float n array):
            Array of velocities in AU/day
        Tp_dim (float n array):
            Array of times in units of days

    """

    pos_I = unitConversion.convertPos_to_dim(pos_R).to('AU')
    C_I2G = inert2geo(t_mjd, t_veq)
    pos_G = C_I2G @ pos_I

    state_EMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', t_mjd)
    posEMB = state_EMB[0].get_xyz().to('AU')
    velEMB = state_EMB[1].get_xyz().to('AU/day')

    posEMB_E = (icrs2gmec(posEMB, t_mjd)).to('AU')

    pos_GCRS = pos_G + posEMB_E  # G frame

    pos_H = (gmec2icrs(pos_GCRS, t_mjd)).to('AU')

    vel_I = rot2inertV(np.array(pos_R), np.array(vel_R), 0)
    v_dim = unitConversion.convertVel_to_dim(vel_I).to('AU/day')
    vel_H = velEMB + v_dim

    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H


def convertSC_I2H(pos_I, vel_I, currentTime, C_I2G, Tp_can=None):
    """Converts initial conditions from the I frame to the H frame

    Args:
        pos_I (float n array):
            Array of distance in canonical units
        vel_I (float n array):
            Array of velocities in canonical units
        currentTime (astropy Time array)
            Current mission time in MJD
        C_I2G (float n array):
            3x3 Array for the directional cosine matrix from the I frame to the G frame
        Tp_can (float n array, optional):
            Optional array of times in canonical units

    Returns:
        tuple:
        pos_H (float n array):
            Array of distance in AU
        vel_H (float n array):
            Array of velocities in AU/day
        Tp_dim (float n array):
            Array of times in units of days

    """

    # Convert to dimensional
    pos_I = unitConversion.convertPos_to_dim(pos_I).to('AU')
    vel_I = unitConversion.convertVel_to_dim(vel_I).to('AU/day')

    # Convert from I frame to (G) frame
    pos_G = C_I2G @ pos_I
    vel_G = C_I2G @ vel_I

    # Get Earth-Moon barycenter in GMEc frame
    posEMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz().to('AU')
    velEMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[1].get_xyz().to('AU/d')
    posEMB_gmec, velEMB_gmec = icrs2gmec(posEMB, currentTime, velEMB)  # km and km/s
    posEMB_gmec = posEMB_gmec.to('AU')
    velEMB_gmec = velEMB_gmec.to('AU/d')

    # Convert data to GMEc frame
    pos_gmec = pos_G + posEMB_gmec
    vel_gmec = vel_G + velEMB_gmec

    # Convert from G frame (AU and AU/d) to H frame (km and km/s)
    pos_H, vel_H = gmec2icrs(pos_gmec, currentTime, vel_gmec)
    pos_H = pos_H.to('AU')
    vel_H = vel_H.to('AU/d')

    if Tp_can is not None:
        Tp_dim = unitConversion.convertTime_to_dim(Tp_can).to('day')
        return pos_H, vel_H, Tp_dim
    else:
        return pos_H, vel_H


def convertSC_H2I(pos_H, vel_H, currentTime, C_I2G, Tp_can=None):
    """Converts initial conditions (or any position and velocity) from the H frame to the I frame

    Args:
        pos_H (float n array):
            Array of distance in canonical units
        vel_H (float n array):
            Array of velocities in canonical units
        currentTime (astropy Time array)
            Current mission time in MJD
        C_I2G (float n array):
            3x3 Array for the directional cosine matrix
        Tp_can (float n array, optional):
            Optional array of times in canonical units

    Returns:
        tuple:
        pos_I (float n array):
            Array of distance in AU
        vel_I (float n array):
            Array of velocities in AU/day
        Tp_dim (float n array):
            Array of times in units of days

    """

    C_G2I = C_I2G.T

    # Convert to dimensional
    pos_H = unitConversion.convertPos_to_dim(pos_H).to('AU')
    vel_H = unitConversion.convertVel_to_dim(vel_H).to('AU/d')

    # Convert given data to GMEc frame
    pos_gmec, vel_gmec = icrs2gmec(pos_H, currentTime, vel_H)  # km and km/s
    pos_gmec = pos_gmec.to('AU')
    vel_gmec = vel_gmec.to('AU/d')

    # Get EM barycenter position in GMEc frame
    posEMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[0].get_xyz().to('AU')
    velEMB = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)[1].get_xyz().to('AU/d')
    posEMB_gmec, velEMB_gmec = icrs2gmec(posEMB, currentTime, velEMB)  # km and km/s
    posEMB_gmec = posEMB_gmec.to('AU')
    velEMB_gmec = velEMB_gmec.to('AU/d')

    # Convert to G frame, then I frame
    pos_G = pos_gmec - posEMB_gmec
    pos_I = C_G2I @ pos_G

    vel_G = vel_gmec - velEMB_gmec
    vel_I = C_G2I @ vel_G

    return pos_I, vel_I


def convertSC_R2I(t_start, t_current, C_I2G, state, mu_star):
    # canonical units
    
    C_I2R = inert2rot(t_current, t_start)
    C_R2I = C_I2R.T
    # Get position of the moon at the epoch in the inertial frame
    sun_I, earth_I, moon_I = getSunEarthMoon(t_current, C_I2G)  # I frame [AU]
    moon_I_can = unitConversion.convertPos_to_canonical(moon_I)

    IC = np.zeros(6)
    
    # Transform position ICs to the epoch moon
    ideal_moon = np.array([1-mu_star, 0, 0])
#    IC[0:3] = C_R2I@(state[0:3] - ideal_moon) + moon_I_can
    IC[0:3] = C_R2I@state[0:3]
    ideal_moon_I = C_R2I@ideal_moon

    # Convert the velocity to I frame from R frame (position is the same in both)
#    vO = rot2inertV(np.array(IC[0:3]), np.array(state[3:6]), 0)
    vO = rot2inertV(np.array(state[0:3]), np.array(state[3:6]), 0)

    # Rotate velocity vector to match the epoch moon (I frame)
    theta = np.arccos((np.dot(moon_I_can, ideal_moon_I))/(np.linalg.norm(moon_I_can)*np.linalg.norm(ideal_moon_I)))
    if theta > np.pi/2:
        theta = -theta
    rot_matrix = rot(theta, 3)
    IC[3:6] = rot_matrix @ vO  # Canonical, I frame

    # Convert IC to dimensional, rotating frame (for STK)
    pos_canrot = C_I2R @ IC[0:3]  # Canonical, R frame
    vel_canrot = inert2rotV(pos_canrot, IC[3:6], 0)  # Canonical, R frame
    pos_dimrot = unitConversion.convertPos_to_dim(pos_canrot).to('km')  # R frame, dimensional
    vel_dimrot = unitConversion.convertVel_to_dim(vel_canrot).to('km/s')  # R frame, dimensional
    print('Dimensional [km] position IC in the rotating frame: ', pos_dimrot)
    print('Dimensional [km/s] velocity IC in the rotating frame: ', vel_dimrot)
#    breakpoint()
    return pos_dimrot, vel_dimrot, IC
    
def convertSC_I2R(t_start, t_current, C_I2G, state, mu_star):
    # canonical units
    
        
    C_I2R = inert2rot(t_current, t_start)
    C_R2I = C_I2R.T
    
    # Get position of the moon at the epoch in the inertial frame
    sun_I, earth_I, moon_I = getSunEarthMoon(t_current, C_I2G)  # I frame [AU]
    moon_I_can = unitConversion.convertPos_to_canonical(moon_I)
    
    # Transform position ICs to the epoch moon
    ideal_moon = np.array([1-mu_star, 0, 0])
#    IC[0:3] = C_R2I@(state[0:3] - ideal_moon) + moon_I_can
    pos = C_R2I@state[0:3]
    ideal_moon_I = C_R2I@ideal_moon
    
    # I frame DCM correction to account for real moon rotated from ICRS vs idealized moon rotated from R
    theta = np.arccos((np.dot(moon_I_can, ideal_moon_I))/(np.linalg.norm(moon_I_can)*np.linalg.norm(ideal_moon_I)))
    if theta > np.pi/2:
        theta = -theta
    C_IR2IH = rot(theta, 3)
    C_IH2IR = C_IR2IH.T
    
    vel_adjI = C_IH2IR@state[3:6]
    vel = inert2rotV(state[0:3], vel_adjI, 0)
    
    return pos, vel

def getSunEarthMoon(currentTime, C_I2G, pvFlag = None):
    """Retrieves the position of the Sun, Earth, and Moon at a given time in AU in the I frame

        Args:
            currentTime (astropy Time array)
                Current mission time in MJD
            C_I2G (float n array):
                3x3 Array for the directional cosine matrix

        Returns:
            r_SunEM (astropy Quantity array):
                Position vector for the Sun in the I frame [AU]
            r_EarthEM (astropy Quantity array):
                Position vector for the Earth in the I frame [AU]
            r_MoonEM (astropy Quantity array):
                Position vector for the Moon in the I frame [AU]

        """

    C_G2I = C_I2G.T

    # Positions of the Sun, Moon, and EM barycenter relative SS barycenter in H frame
    rv_Sun = get_body_barycentric_posvel('Sun', currentTime)
#    rv_Earth = get_body_barycentric_posvel('Earth', currentTime)
    rv_Moon = get_body_barycentric_posvel('Moon', currentTime)
    rv_EM = get_body_barycentric_posvel('Earth-Moon-Barycenter', currentTime)
    
    r_SunO = rv_Sun[0].get_xyz()
#    r_EarthO = rv_Earth[0].get_xyz()
    r_MoonO = rv_Moon[0].get_xyz()
    r_EMO = rv_EM[0].get_xyz()

    # Convert from H frame (AU) to GMEc frame (km)
    if pvFlag:
        r_SunGMEc, v_SunGMEc = icrs2gmec(r_SunO, currentTime, v_SunO)
        r_MoonGMEc, v_MoonGMEc = icrs2gmec(r_MoonO, currentTime, v_MoonO)
        r_EMGMEc, v_EMGMEc = icrs2gmec(r_EMO, currentTime, v_EMO)
    else:
        r_SunGMEc = icrs2gmec(r_SunO, currentTime)
        r_MoonGMEc = icrs2gmec(r_MoonO, currentTime)
        r_EMGMEc = icrs2gmec(r_EMO, currentTime)
        
    # Convert from GMEc frame to G frame (change origin to EM barycenter)
    r_SunG = r_SunGMEc - r_EMGMEc
    r_EarthG = -r_EMGMEc
    r_MoonG = r_MoonGMEc - r_EMGMEc

    # Convert from G frame (in km) to I frame (in AU)
    r_SunEM = C_G2I @ r_SunG.to_value(u.AU)
    r_EarthEM = C_G2I @ r_EarthG.to_value(u.AU)
    r_MoonEM = C_G2I @ r_MoonG.to_value(u.AU)

    if pvFlag:
        v_SunO = rv_Sun[1].get_xyz()
#        v_EarthO = rv_Earth[1].get_xyz()
        v_MoonO = rv_Moon[1].get_xyz()
        v_EMO = rv_EM[1].get_xyz()
    
        v_SunG = v_SunGMEc - v_EMGMEc
        v_EarthG = -v_EMGMEc
        v_MoonG = v_MoonGMEc - v_EMGMEc
    
        v_SunEM = C_G2I @ v_SunG.to_value(u.AU/u.d)
        v_EarthEM = C_G2I @ v_EarthG.to_value(u.AU/u.d)
        v_MoonEM = C_G2I @ v_MoonG.to_value(u.AU/u.d)

        return r_SunEM, r_EarthEM, r_MoonEM, v_SunEM, v_EarthEM, v_MoonEM
    else:
        return r_SunEM, r_EarthEM, r_MoonEM

