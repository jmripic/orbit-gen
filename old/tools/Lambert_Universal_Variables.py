"""
This function will return the initial and final velocities of a 2-body orbit given an
initial and final position and time of flight. This code assumes a single central body
and will return the outputs in any units that are used for the inputs. This algorithm is 
based on the Universal Variable Method presented in Vallado's Fundamentals of 
Astrodynamics and Applications. This python file is adapted by Colby C. Merrill from a 
MATLAB file initially written by Rodney L. Anderson.

The inputs to the function are
	r0: 3x1 array; the cartesian components of the spacecraft's initial position
	rf: 3x1 array; the cartesian components of the spacecraft's final position
	TOF: scalar; the time of flight of the spacecraft
	dM: scalar; must be set to 1 (short way transfer) or -1 (long way transfer)
	mu: scalar; the gravitational parameter of the system
The outputs of the function are
	v0: 3x1 array; the cartesian components of the initial velocity vector
	vf: 3x1 array; the cartesian components of the final velocity vector
"""

import numpy as np

def Lambert(r0, rf, TOF, dM, mu):

	r0mag = np.linalg.norm(r0)
	rfmag = np.linalg.norm(rf)

	cosv = np.dot(r0, rf)/(r0mag*rfmag)
	deltav = np.arccos(np.radians(cosv))
	A = dM*np.sqrt(r0mag*rfmag*(1+cosv))

	if deltav == 0:
		A = 0 #This is an edge case and an error
	else:
		psi = 0
		C2 = 0.5
		C3 = 1/6
		psiup = 4*np.pi**2
		psilow = -4*np.pi
	
	deltat = 0
	count = 1

	while np.absolute(TOF - deltat) > 1e-6 and count < 1000:
		y = r0mag + rfmag + (A*(psi*C3 - 1))/np.sqrt(C2)
		if y < 0:
			psilow += 1
		X = np.sqrt(y/C2)
		deltat = (X**3*C3 + A*np.sqrt(y))/np.sqrt(mu)
		if deltat < TOF:
			psilow = psi
		else:
			psiup = psi
		psi = (psiup + psilow)/2
		if psi > 1e-6:
			C2 = (1 - np.cos(np.sqrt(psi)))/psi
			C3 = (np.sqrt(psi) - np.sin(np.sqrt(psi)))/np.sqrt(psi**3)
		elif psi < -1e-6:
			C2 = (1 - np.cosh(np.sqrt(-psi)))/psi
			C3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi))/np.sqrt((-psi)**3)
		else:
			C2 = 1/2
			C3 = 1/6
		count += 1

	f = 1 - y/r0mag
	g = A*np.sqrt(y/mu)
	gdot = 1 - y/rfmag

	v0 = (rf - np.multiply(f,r0))/g
	vf = (gdot*rf - r0)/g

	return v0, vf

# Example Case
mu = 1.32712440018E+20*(1/149597870700)**3*(86400)**2 #(AU^3/day^2)

r0 = np.array([-1.549855031371103E-01, -1.003565367649044E+00, 8.070604616748505E-05]) #AU
rf = np.array([-7.211816063071923E-01, 1.096664583204050E+00, -8.130232078400046E-03]) #AU
TOF = 314.66 #days
DM = -1

[v0,vf] = Lambert(r0,rf,TOF,DM,mu)