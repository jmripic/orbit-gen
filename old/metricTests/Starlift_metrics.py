
# This file contains the class orbit_eval. Objects of the class are orbit_pickle_files that have been
# loaded using the load_pickle_file function included below. 
# orbit_eval contains methods that evaluate a spacecraft throughout its orbit from the standpoint
# of different metrics. 
# Its orbit_volume function passes all test cases as of 5/17/24. It was verified for cases in which
# the spacecraft stayed above the specified altitude for the duration of its orbit. The method is 
# rather lengthy because different scenarios require very different calculations of volume.
# As of 5/17/24, the angular_diameter function also passes all test cases.

import sys
sys.path.insert(1, sys.path[0][0:-11])

# from tools.Solution import Solution as Solution
import numpy as np
import matplotlib.pyplot as plt
# from astropy import constants
from scipy import integrate as int
import math
import pickle
import os.path



# establish constants
M_m = 7.349*(10)**22 # mass of moon
R_m = 1737400 # radius of moon
# M_e = constants.M_earth.value # 5.97219*(10)**24 # mass of earth
M_e = 5.97219*(10)**24 # mass of earth
d = (3.8444*10**5)*1000 # distance between earth and moon in m
# G = constants.G.value  # 6.6743*10**(-11) # Gravitational constant
G = 6.6743*10**(-11) # Gravitational constant
x_com = (M_m*d) / (M_m + M_e) # center of mass position
m_position = d - x_com # moon distance from barycenter


def load_pickle_file(path_str):
  # this function takes a string specifying the path of a pickle file and loads it into an orbit variable
  path_f1 = os.path.normpath(os.path.expandvars(path_str))
  f1 = open(path_f1, "rb")
  orbit = pickle.load(f1)
  f1.close()

  return(orbit)

# load pickle files below
DRO_11 = load_pickle_file("orbitFiles/DRO_11.241_days.p")
DRO_13 = load_pickle_file("orbitFiles/DRO_13.0486_days.p")
L2_12 = load_pickle_file("orbitFiles/L2_S_12.05_days.p")


class orbit_eval:
  def __init__(self,orbit):
    self.orbit = orbit # orbit is a pickle file that has been loaded using the load_pickle_file function above

  def orbit_volume(self,angle,A,plots):
    # evaluates how well the spacecraft views volume of moon below certain orbit
    # angle is the scope of the telescope in degrees (ex: 2 deg by 2 deg means angle)
    # A is the altitude of orbits you are considering in meters
    # plots: type 'Y' if you want plots immediately and 'N' if you do not want plots
    # time is a time array
    # V_percent_viewed is an array containing the volume seen by the spacecraft an any time in the orbit for
    # each time in the time array
    # r_mag_vec gives the distance of the spacecraft away from the moon at all times in the time array


    # state = self.orbit['state'] # gather position and velocity data
    state = self.orbit['state']
    dimensions = np.shape(state)
    rows, columns = dimensions # obtain size of state
    r_satellite = state[:,0:3]*1000
    time = self.orbit['t'] # get time
    phi = (angle) * np.pi / 180 / 2
    Va = (4/3)*np.pi*(R_m+A)**3

    # move positions relative to the moon
    r_moon = np.column_stack((np.ones((rows,1))*m_position, np.zeros((rows,2))))
    r = r_satellite - r_moon #  array describing satellite distance from moon

    V = []
    r_mag_vec = []

    for i in range(0,rows):
      r_mag = (r[i,0]**2 + r[i,1]**2 + r[i,2]**2)**(1/2)
      r_mag_vec.append(r_mag)

      if r_mag < R_m + A:
        # Satellite is within max altitude
        if r_mag*np.sin(phi) < R_m:
          
          # scenario 1 for within altitude
          alpha_fake = np.arcsin(np.sin(phi)*r_mag/R_m)
          alpha = np.pi - alpha_fake
          theta = np.pi - alpha - phi
          L = R_m*np.sin(theta)
          h = r_mag - R_m*np.cos(theta)
          V_cone = (np.pi/3)*(L**2)*h
          V_cap = (np.pi/3)*(R_m**3)*(2+np.cos(theta))*(1-np.cos(theta))**2

          scenario = '1within'
          volume_viewed = V_cone - V_cap

        else:
          # scenario 2 for within altitude
          alpha = np.arcsin((np.sin(phi)/(R_m+A))*r_mag)
          theta = np.pi - alpha - phi

          if theta < (np.pi/2):
            tau = np.arccos(R_m / (R_m + A))
            omega = np.arccos(R_m/r_mag)
            gamma = tau+omega

            if gamma < (np.pi/2):
              # scenario 2a for within altitude
              V_frontcapsmall = (np.pi/3)*(R_m+A)**3 * (2+np.cos(theta))*(1-np.cos(theta))**2
              V_frontcapbig = (np.pi/3)*(R_m+A)**3 * (2+np.cos(gamma))*(1-np.cos(gamma))**2
              b = (R_m+A)*np.sin(theta)
              a = (R_m+A)*np.sin(theta)
              h = (R_m+A)*(np.cos(theta) - np.cos(gamma))
              V_conical = (np.pi/3)*h*(a**2 + a*b + b**2)
              V_s2 = V_frontcapbig - V_frontcapsmall - V_conical

              L = R_m*np.sin(omega)
              h = r - R_m*np.cos(omega)
              V_cone = (np.pi/3)*(L**2)*h
              V_cap = (np.pi/3)*(R_m)**3*(2+np.cos(omega))*(1-np.cos(omega))**2
              V_s1 = V_cone - V_cap

              delta = (np.pi/2) - omega
              S = np.sin(theta)*(R_m+A) / np.sin(phi)
              L_big = S*np.sin(phi)
              h_big = S*np.cos(phi)
              V_bigcone = (np.pi/3)*(L_big**2)*h_big
              L_small = h_big(np.tan(phi))
              V_smallcone = (np.pi/3)*(L_small**2)*h_big
              V_s3 = V_bigcone - V_smallcone

              scenario = '2within'
              volume_viewed = V_s1 + V_s2 + V_s3

            else: 
              # scenario 2b for within altitude
              alpha = np.arcsin(np.sin(phi)*r/(R_m+A))
              omega = np.arccos(R_m/r_mag)
              delta = np.pi/2 - omega
              tau = theta - omega
              q = np.arcsin(R_m / (R_m+A))
              f = np.pi/2 - q
              u = np.pi - omega - f
              a = (R_m+A)*np.sin(u)
              h_small = (R_m+A)*np.sin(theta)
              h_big = r_mag - h_small
              b = h_big*np.tan(delta)
              
              V_frontcap = (np.pi/3)*(R_m+A)**3 * (2+np.cos(theta))*(1-np.cos(theta))**2
              V_backcap = (np.pi/3)*(R_m+A)**3 * (2+np.cos(u))*(1-np.cos(u))**2

              h = (R_m + A)*np.cos(theta) + (R_m+A)*np.cos(u)
              V_conical = (np.pi/3)*h*(a**2 + a*b + b**2)
              Vs3 = (4/3)*np.pi(R_m+A)**3 - V_frontcap - V_backcap - V_conical

              L = (R_m + A)*np.sin(theta)
              Vs2 = (np.pi/3)*(L**2)*h_big - (np.pi/3)*(b**2)*h_big

              omega = np.arccos(R_m/(R_m+A))
              L = R_m*np.sin(omega)
              h = r_mag - R_m*np.cos(omega)
              V_cone = (np.pi/3)*L**2*h
              V_cap = (np.pi/3)*(R_m)**3*(2+np.cos(omega))*(1-np.cos(omega))**2
              Vs1 = V_cone - V_cap

              scenario = '2within'
              volume_viewed = Vs1 + Vs2 + Vs3
          
          else:
            # SCENARIO 2C for within altitude
            delta = np.arctan(R_m/r_mag)
            omega = (np.pi/2) - delta
            tau = np.arccos(R_m / (R_m+A))
            gamma = np.pi - tau - omega

            alpha = np.arcsin(np.sin(phi)*r_mag / (R_m + A))
            theta = np.pi - alpha - phi
            beta = np.pi - theta - gamma
            
            V_bigcap = (np.pi/3)*(R_m+A)**3 *(2+np.cos(beta+gamma))*(1-np.cos(beta+gamma))**2
            V_smallcap = (np.pi/3)*(R_m+A)**3 * (2+np.cos(gamma))*(1-np.cos(gamma))**2

            L1 = (R_m + A)*np.sin(beta + gamma)
            d = (R_m + A)*np.cos(beta + gamma)
            a = (R_m + A)*np.sin(gamma)
            b = (r_mag + d)*np.tan(delta)
            h1 = (R_m+A)*np.cos(gamma) - d
            V_conical = (np.pi/3)*h1*(a**2 + a*b + b**2)

            Vs3 = V_bigcap - V_smallcap - V_conical

            V_bigcone = (np.pi/3)*(L1**2)*(r_mag+d)
            V_smallcone = (np.pi/3)*(b**2)*(r_mag+d)

            Vs2 = V_bigcone - V_smallcone

            L = R_m*np.sin(omega)
            h = r_mag - R_m*np.cos(omega)
            V_cone = (np.pi/3)*L**2*h
            V_cap = (np.pi/3)*(R_m)**3*(2+np.cos(omega))*(1-np.cos(omega))**2

            Vs1 = V_cone - V_cap

            scenario = '2within'
            volume_viewed = Vs1 + Vs2 + Vs3
      
      else:
        # OUTSIDE ALTITUDE
        if r_mag*np.sin(phi) < R_m:
          # scenario 1
          alpha_mfake = np.arcsin(np.sin(phi)*r_mag/R_m)
          alpha_afake = np.arcsin(np.sin(phi)*r_mag/(R_m+A))

          alpha_m = np.pi - alpha_mfake
          alpha_a = np.pi - alpha_afake

          theta_m = np.pi - phi - alpha_m
          theta_a = np.pi - phi - alpha_a

          V_cap_m = (np.pi/3)*(R_m)**3 * (2+np.cos(theta_m))*(1-np.cos(theta_m))**2
          V_cap_a = (np.pi/3)*(R_m+A)**3 * (2+np.cos(theta_a))*(1-np.cos(theta_a))**2

          a = R_m*np.sin(theta_m)
          b = (R_m + A)*np.sin(theta_a)

          h = (R_m+A)*np.cos(theta_a) - R_m*np.cos(theta_m)
          V_conical = (np.pi/3)*h*(a**2 + a*b + b**2)

          scenario = '1Outside'
          volume_viewed = V_cap_a + V_conical - V_cap_m

        elif (r_mag*np.sin(phi) < R_m + A):
          
          # scenario 2, in between case where view spans moon but not R_m+A
          # find spherical cap volume in front
          gamma = np.arcsin(R_m/r_mag)
          delta = (np.pi/2) - gamma
          V_front = (np.pi/3)*(R_m**3)*(2+np.cos(delta))*(1-np.cos(delta))**2

          # find cone section volume
          b = R_m*np.sin(delta)
          omega = np.arccos(R_m/(R_m+A))
          alpha = np.pi - delta - omega
          a = (R_m+A)*np.sin(alpha)
          h = R_m*np.cos(delta) + (R_m+A)*np.cos(alpha)
          V_cone = (np.pi/3)*h*(a**2 + a*b + b**2)

          # find back-spherical cap volume
          V_back = (np.pi/3)*((R_m+A)**3)*(2+np.cos(alpha))*(1-np.cos(alpha))**2

          sigma = np.arcsin((r_mag*np.sin(phi)) / (R_m + A))
          sigma = np.pi - sigma # correction of alpha to account for arcsin
          tao = np.pi - phi - sigma
          epsilon = 2*sigma - np.pi

          if (tao + epsilon) > (np.pi/2):

            # find spherical cap information
            zeta = np.pi - tao - epsilon
            V_spherical = ((np.pi/3)*(R_m+A)**3)*(4-(2+np.cos(tao))*(1-np.cos(tao))**2 - (2+np.cos(zeta))*(1-np.cos(zeta))**2)

            # find partial cone information
            b = (R_m+A)*np.sin(tao)
            a = (R_m+A)*np.sin(zeta)
            h = (R_m+A)*(np.cos(tao)+np.cos(zeta))
            V_conical = (np.pi/3)*h*(a**2 + a*b + b**2)

            V_slit = V_spherical - V_conical # volume of space between cone section and sphere section

            scenario = '2outside'
            volume_viewed = Va - V_slit - V_front - V_cone - V_back

          else:
            # find spherical cap information
            V_spherical = ((np.pi/3)*(R_m+A)**3)*((2+np.cos(epsilon+tao))*(1-np.cos(epsilon+tao))**2 - (2+np.cos(tao))*(1-np.cos(tao))**2)

            # find cone information
            b = (R_m+A)*np.sin(tao)
            a = (R_m+A)*np.sin(epsilon+tao)
            h = (R_m+A)*(np.cos(tao) - np.cos(epsilon+tao))
            V_conical = (np.pi/3)*h*(a**2 + a*b + b**2)

            V_slit = V_spherical - V_conical # volume of space between cone section and sphere section

            scenario = '2outside'
            volume_viewed = Va - V_slit - V_front - V_cone - V_back

        else:
          # scenario 3, view spans entire moon and altittude
          gamma = np.arcsin(R_m/r_mag)
          delta = (np.pi/2) - gamma
          V_front = (np.pi/3)*(R_m**3)*(2+np.cos(delta))*(1-np.cos(delta))**2 # spherical cap volume in front

          # find cone section volume
          b = R_m*np.sin(delta)
          omega = np.arccos(R_m/(R_m+A))
          alpha = np.pi - delta - omega
          a = (R_m+A)*np.sin(alpha)
          h = R_m*np.cos(delta) + (R_m+A)*np.cos(alpha)
          V_cone = (np.pi/3)*h*(a**2 + a*b + b**2)

          # find back-spherical cap volume
          V_back = (np.pi/3)*((R_m+A)**3)*(2+np.cos(alpha))*(1-np.cos(alpha))**2

          volume_viewed = Va - V_front - V_cone - V_back
      
      Volume = ((4/3)*np.pi)*((R_m+A)**3 - (R_m)**3)
      V.append(volume_viewed)
    
    Volume = ((4/3)*np.pi)*((R_m+A)**3 - (R_m)**3)
    V_percent_viewed = []

    for i in range(rows):
      V_percent_viewed.append(V[i] / Volume)
      #if i == 180:
        #print('Fraction of Volume: ' + str(V[i] / Volume))
      
    if plots == 'Y': # volume plot included if specified in function call
      plt.figure()
      plt.plot(time,V_percent_viewed,label = 'Fraction of Volume')
      plt.xlabel("Time (s)")
      plt.ylabel("Visible fraction of Orbit Volume")
      plt.title("Spacecraft in Orbit Evaluation")
      plt.legend()
      plt.show()

    return time,V_percent_viewed,r_mag_vec

  def angular_diameter(self,angle,A,plots):
    # simiilar analysis as volume, but with angular diameter
    # angle is the field of view
    # A is the altitude of orbits
    # plots: type 'Y' if you want plots immediately and 'N' if you do not want plots
    # time is a time array
    # ang_diam_frac is a list containing the fraction of the angular diameter of the moon + altitude you can see at any point in time

    state = self.orbit['state'] # gather position and velocity data
    #state = self.orbit.statevec
    rows, columns = np.shape(state) # obtain size of state
    r_satellite = state[:,0:3]*1000
    #time = self.orbit.tvec #
    time = self.orbit['t'] # get time
    phi = (angle) * np.pi / 180 / 2  # half-angle of aperture in radians
    d = 2*(R_m + A) # diameter of moon + altitude

    fixed_diam = phi*2 # angular diameter of view phi*2
    ang_diam = []
    ang_diam_frac = []

    r_moon = np.column_stack((np.ones((rows,1))*m_position, np.zeros((rows,2))))
    r = r_satellite - r_moon #  array describing satellite distance from moon

    V = []
    r_mag_vec = []
    
    for i in range(0,rows):
      r_mag = (r[i,0]**2 + r[i,1]**2 + r[i,2]**2)**(1/2)
      a_diam = 2*np.arcsin(d/(2*r_mag))
      a_diam_frac = fixed_diam / a_diam
      r_mag_vec.append(r_mag)
      ang_diam.append(a_diam)

      if a_diam <= fixed_diam: # object + altitude is in full view
        ang_diam_frac.append(1)
      else:
        ang_diam_frac.append(a_diam_frac)
    
    if plots == 'Y': # angular diameter plot included if specified in function call
      plt.figure()
      plt.plot(time,ang_diam_frac,label = 'Fraction of Angular Diameter')
      plt.xlabel("Time (s)")
      plt.ylabel("Visible fraction of Orbit Angular Diameter")
      plt.title("Spacecraft in Orbit Evaluation")
      plt.legend() 
      plt.show()

    return time,ang_diam_frac
  
  def solid_angle(self,angle,A,plots):
    # volume solution using steradians
    # angle is the field of view
    # A is the altitude of orbits
    # ster_vis is a list containing the visible solid angle of the moon + altitude at all points in time
    # ster_frac is a list containing the fraction of the solid angle of the moon + altitude you can see at any point in time

    state = self.orbit['state'] # gather position and velocity data
    # state = self.orbit.statevec
    rows, columns = np.shape(state) # obtain size of state
    r_satellite = state[:,0:3]*1000
    # time = self.orbit.tvec # 
    time = self.orbit['t'] # get time
    phi = (angle) * np.pi / 180 / 2  # half-angle of aperture in radians
    R = (R_m + A) # diameter of moon + altitude

    ster_vis = []
    ster_frac = []

    r_moon = np.column_stack((np.ones((rows,1))*m_position, np.zeros((rows,2))))
    r = r_satellite - r_moon #  array describing satellite distance from moon

    r_mag_vec = np.linalg.norm(r, axis=1).reshape(rows,1)

    for i in range(0,rows):
      r_mag = r_mag_vec[i,0]
      stermax = 2*np.pi*(1-np.sqrt(r_mag**2-R**2)/r_mag)
      visrad = r_mag*np.tan(phi)
      stervis = 2*np.pi*(1-np.sqrt(r_mag**2-visrad**2)/r_mag)
      sterfraction = stervis / stermax 
      ster_vis.append(stervis)

      if stervis >= stermax: # object + altitude is in full view
        ster_frac.append(1)
      else:
        ster_frac.append(sterfraction)

    if plots == 'Y': # solid angle plot included if specified in function call
      plt.figure()
      plt.plot(time,ster_frac,label = 'Fraction of Visible Solid Angle')
      plt.xlabel("Time (s)")
      plt.ylabel("Visible fraction of Orbit Volume")
      plt.title("Spacecraft in Orbit Evaluation")
      plt.legend() 
      plt.show()

    return ster_vis ,ster_frac
  
  def orbit_ranker(self,orbit_files,orbit_names,angle,A,metric):
    # orbit_files is a list of orbit pickle files
    # orbit_names is the names of the orbits
    # angle is the filed of view
    # A is the altitude
    # metric is the metric by which the orbits are evaluated, which should be a string that is the same as the name of the function
    # avg is a numpy array containing the average value for each orbit in the order they were inputted
    # indices is the indices of the ranking based on the order they were inputted
    # ranking will output the names of the orbits, first name has the lowest avg and last name has the highest

    avg = []

    for i in range(0,len(orbit_files)):

      if metric == "orbit_volume":
        orbit = orbit_files[i]
        [time,values,r_mag_vec] = orbit.orbit_volume(angle,A,'N') # obtain volumes for metric throughout orbit

      elif metric == "angular_diameter":
        orbit = orbit_files[i]
        [time,values] = orbit.angular_diameter(angle,A,'N') # obtain angular diameter for metric throughout orbit

      # convert time from pickle file to 1D array
      timevec = []
      for j in range(1,len(time)):
        timevec.append(time[j][0]) 
      time = timevec

      # obtain time vector of equally-spaced points
      total_points = math.ceil(time[len(time)-1] / (time[1] - time[0]))
      end_time = time[len(time)-1]
      time = np.array(time) 

      # create corresponding values for metric for each time using interpolation
      values = np.array(values[0:(len(values)-1)])
      tvals = np.linspace(0,end_time,total_points+10)
      yInt = np.interp(tvals,time,values)

      # find average of metric
      yInt = np.array(yInt)
      avg.append(np.mean(yInt))

    avg = np.array(avg) # convert list to numpy array

    # establish ranking (updated)
    indices = np.argsort(avg)
    ranking = np.empty(len(indices), dtype=object)

    for i in range(len(indices)):
      index = indices[i]
      ranking[i] = orbit_names[index]
      
    return avg,indices,ranking

DRO_11 = orbit_eval(DRO_11)
DRO_13 = orbit_eval(DRO_13)
L2_12 = orbit_eval(L2_12)



angle = 3 # width of view in degrees
A = 100000
[time11,V_percent_viewed11,r_mag_vec] = DRO_11.orbit_volume(angle,A,'N')
[time,ang_diam_frac11] = DRO_11.angular_diameter(angle,A,'N')
[time,ang_diam_frac] = DRO_11.solid_angle(angle,A,'N')

[time13,V_percent_viewed13,r_mag_vec] = DRO_13.orbit_volume(angle,A,'N')
[time12,V_percent_viewed12,r_mag_vec] = L2_12.orbit_volume(angle,A,'N')

orbits_files = [DRO_11,DRO_13,L2_12]
orbit_names = ["DRO_11","DRO_13","L2_12"]

# plt.plot(time11,V_percent_viewed11,label = 'volume fraction')
# plt.plot(time11,ang_diam_frac11,label = 'angular diameter fraction')
# plt.xlabel("Time (s)")
# plt.ylabel("Fraction")
# plt.title("Spacecraft in Orbit Evaluation")
# plt.legend()

# plt.show()


[avg,indices,ranking] = DRO_11.orbit_ranker(orbits_files,orbit_names,angle,A,"orbit_volume")
print(ranking)
print(avg)
print(indices)

plt.plot(time11,V_percent_viewed11,label = 'DRO 11')
plt.plot(time13,V_percent_viewed13,label = 'DRO 13')
plt.plot(time12,V_percent_viewed12,label = 'L2 12')

plt.xlabel("Time (s)")
plt.ylabel("Fraction of Volume Viewed")
plt.title("Spacecraft in Orbit Evaluation")
plt.legend()

plt.show()
