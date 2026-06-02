# This file contains a class that will be used to develop transfer orbits.
# As of 5/17/24 the class contains functions that can determine a transfer orbit using lambert's problem
# and graph them. This function is only verified for 1 case, where r1 = [0,7500000], r2 = [9000000,0],
# and the semi-major axis of the transfer is 10000000 meters


import numpy as np
import math
import matplotlib.pyplot as plt

pi = math.pi
mu = 3.986*10**14

class transfer_orbits:
    def __init__(self,orbit):
        self.orbit = orbit # orbit is a pickle file
        #self.SMA1 = SMA1 # semimajor axis of 1
        #self.SMA2 = SMA2 # semimajor axis of 2
        #self.smia1 = smia1 # semiminor axis of 1
        #self.smia2 = smia2 # semiminor axis of 2
        #self.tilt1 = tilt1 # angle of tilt from axis of 1
        #self.tilt2 = tilt2 # angle of tilt from axis of 2


    def lambert(self,r1,r2,a):
        # r1 is 2-element numpy array describing location at time of departure
        # r2 is 2-element numpy array describing location at time of arrival
        # a is semi-major axis

        # returns geometric elements needed to describe elliptical transfer orbits (TO's)
        # b1 and b2 are semi-minor axes of TO's
        # phi1 and phi2 are angles centers of ellipse make with x-axis
        # center1 and center2 are lists containing the coordinates of the center of the TO's

        x1 = r1[0] # x-coordinate at time of departure
        y1 = r1[1] # y-coordinate at time of departure

        x2 = r2[0] # x-coordinate at time of arrival
        y2 = r2[1] # y-coordinate at time of arrival

        r1mag = math.sqrt(x1**2 + y1**2) # distance from planet to departure point
        r2mag = math.sqrt(x2**2 + y2**2) # distance from planet to arrival point
        r12mag = math.sqrt((x1-x2)**2 + (y1-y2)**2) # distance between departure and arrival points
        nu = math.acos((x1*x2 + y2*y2) / (r1mag*r2mag)) # angle between arrival and departure vectors

        s = (r1mag + r2mag + r12mag) / 2
        alpha = 2*math.asin(math.sqrt(s/(2*a)))
        beta = 2*math.asin(math.sqrt((s-r12mag)/(2*a)))

        # a_min = s / 2 # semi-major axis of minimum energy TO
        # l_min = (r1mag*r2mag/r12mag)*(1-math.cos(nu)) # l for minimum energy TO
        # e_min = math.sqrt(1-(2*l_min/s)) # eccentricity of minimum energy TO


        R1 = 2*a - math.sqrt(x1**2 + y1**2) # radius of intersecting circle centered at P1
        R2 = 2*a - math.sqrt(x2**2 + y2**2) # radius of intersecting circle centered at P2


        R = math.sqrt((x1-x2)**2 + (y1-y2)**2)

        # make pre-calculaitons for faster speed
        pre_calc1 = (R1**2-R2**2)/(2*R**2)
        pre_calc2 = (1/2)*math.sqrt(2*(R1**2+R2**2)/R**2 - (R1**2-R2**2)**2/R**4 - 1)
        pre_calc3x = (1/2)*(x1+x2) + pre_calc1*(x2-x1)
        pre_calc3y = (1/2)*(y1+y2) + pre_calc1*(y2-y1)

        # coordinates for two possible vacant foci are shown belows
        x_intersect1 = pre_calc3x + pre_calc2*(y2-y1)
        y_intersect1 = pre_calc3y + pre_calc2*(x1-x2)

        x_intersect2 = pre_calc3x - pre_calc2*(y2-y1)
        y_intersect2 = pre_calc3y - pre_calc2*(x1-x2)

        FF1 = math.sqrt(x_intersect1**2 + y_intersect1**2) # distance to vacant focus 1
        FF2 = math.sqrt(x_intersect2**2 + y_intersect2**2) # distance to vacant focus 2

        center1 = [x_intersect1/2,y_intersect1/2] # center of first ellipse
        center2 = [x_intersect2/2,y_intersect2/2] # center of second ellipse


        e1 = FF1 / (2*a) # eccentricity of TO1
        e2 = FF2 / (2*a) # eccentricity of TO2

        c1 = e1*a
        c2 = e2*a

        b1 = math.sqrt(a**2 - c1**2) # semi-minor axis of TO1
        b2 = math.sqrt(a**2 - c2**2) # semi-minor axis of TO2

        Tp = (2*pi)*math.sqrt(a**3/mu)
        T1 = (Tp/(2*pi))*((alpha-math.sin(alpha)) - (beta-math.sin(beta))) # time to intercept for TO1, short way
        T2 = (Tp/(2*pi))*((alpha-math.sin(alpha)) + (beta-math.sin(beta))) # time to intercept for T01, long way

        if x_intersect1 >= 0:
            phi1 = math.atan(y_intersect1/x_intersect1) # angle for tilted elliptical TO1
        else:
            phi1 = pi + math.atan(y_intersect1/x_intersect1)

        if x_intersect2 >= 0:
            phi2 = math.atan(y_intersect2/x_intersect2) # angle for tilted elliptical TO2
        else:
            phi2 = pi + math.atan(y_intersect2/x_intersect2)

        v1mag = math.sqrt(mu*(2/r1mag-1/a))

        return b1,b2,phi1,phi2,center1,center2,T1,T2,v1mag
    
    def orbit_vectors(self,a,b,phi,h,k):
        # gives vector describing elliptical orbit that can be plotted
        # a is semi-major axis
        # b is semi-minor axis
        # phi is angle of tilt
        # (h,k) is center of ellipse
        # returns numpy arrays xpos and ypos, which can be plotted with plt.plot(xpos,ypos)
        theta = np.linspace(0,2*pi,630)

        xpos = a*np.cos(theta)
        ypos = b*np.sin(theta)

        # rotate ellipse by angle phi from x-axis
        new_xpos = xpos*np.cos(phi)-ypos*np.sin(phi)
        new_ypos = xpos*np.sin(phi)+ypos*np.cos(phi)

        # shift ellipse to new center
        for i in range(0,len(new_xpos)):
            new_xpos[i] = new_xpos[i] + h
            new_ypos[i] = new_ypos[i] + k

        return new_xpos,new_ypos
    
    def orbit_plot(self,orbits):
        # orbit vector
        # plot the orbit trajectories
        # first orbit in orbits should be first orbit
        # second orbit in orbits should orbit of target
        # third orbit should be transfer orbit

        for i in range(0,len(orbits)):
            if i < 2:
                plt.plot(orbits[i][0],orbits[i][1]) # plot initial and final orbits
            else:
                plt.plot(orbits[i][0],orbits[i][1], linestyle = '--') # plot transfer orbit

        plt.xlabel('x-position [m]')
        plt.ylabel('y-position [m]')
        plt.legend(["Initial orbit","Orbit of target","Transfer orbit"],loc="upper right")
        plt.axhline(y = 0.5, color = 'r', linestyle = '-') 
        plt.axvline(x = 0.5, color = 'r', linestyle = '-') 
        plt.title("Transfer Orbit via Lambert's Solution")
        plt.axis('equal')
        plt.show()

        return



ch = 5
flight = transfer_orbits(ch)   


r1 = [0,7500000]
r2 = [9000000,0]
a = 10000000

[b1,b2,phi1,phi2,center1,center2,T1,T2,v1mag] = flight.lambert(r1,r2,a)
h = center1[0]
k = center1[1]


[xpos,ypos] = flight.orbit_vectors(a,b1,phi1,h,k)
orbit1 = transfer_orbits(ch)
[xo1,yo1] = flight.orbit_vectors(r1[1],r1[1],0,0,0)
[xo2,yo2] = flight.orbit_vectors(r2[0],r2[0],0,0,0)

orbits = [[xo1,yo1],[xo2,yo2],[xpos,ypos]]

flight.orbit_plot(orbits)

