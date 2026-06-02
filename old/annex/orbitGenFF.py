import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys
import rebound
import matplotlib.pyplot as plt

sys.path.insert(1, '/Users/gracegenszler/Documents/Research/starlift')
import tools

import pdb

# TU = 27.321582 d
# DU = 384400 km
# m_moon = 7.349x10**22 kg
# m_earth = 5.97219x10**24 kg

def CRTBP_EOM(t,w,mu_star):
    """Equations of motion for the CRTBP in the rotating frame

    Args:
        w (float):
            State in non dimensional units [position, velocity]
        mu_star (float):
            Non-dimensional mass parameter

    Returns:
        ~numpy.ndarray(float):
            Time derivative of the state [velocity, acceleration]

    """
    [x,y,z,vx,vy,vz] = w
    
    m1 = 1 - mu_star
    m2 = mu_star

    r1 = mu_star
    r2 = 1 - mu_star
    r_1O_H = r1*np.array([-1,0,0])
    r_2O_H = r2*np.array([1,0,0])

    r_PO_H = np.array([x,y,z])
    v_PO_H = np.array([vx,vy,vz])

    r_P1_H = r_PO_H - r_1O_H
    r_P2_H = r_PO_H - r_2O_H
    r1_mag = np.linalg.norm(r_P1_H)
    r2_mag = np.linalg.norm(r_P2_H)

    F_g1 = -m1/r1_mag**3*r_P1_H
    F_g2 = -m2/r2_mag**3*r_P2_H
    F_g = F_g1 + F_g2

    e3_hat = np.array([0,0,1])

    a_PO_H = F_g - 2*np.cross(e3_hat,v_PO_H) - np.cross(e3_hat,np.cross(e3_hat,r_PO_H))
    ax = a_PO_H[0]
    ay = a_PO_H[1]
    az = a_PO_H[2]

    dw = [vx,vy,vz,ax,ay,az]
    return dw

def calcMonodromyMatrix(freeVar,mu_star,m1,m2):
    """Calculates the monodromy matrix

    Args:
        pos (float):
            Position in non dimensional units
        mu_star (float):
            Non-dimensional mass parameter
        m1 (float):
            Mass of the larger primary in non dimensional units
        m2 (float):
            Mass of the smaller primary in non dimensional units

    Returns:
        ~numpy.ndarray(float):
            monodromy matrix

    """
    x = freeVar[0]
    y = freeVar[1]
    z = freeVar[2]

    r_P0 = np.array([x, y, z])
    r_m10 = -np.array([mu_star, 0, 0])
    r_m20 = np.array([(1 - mu_star), 0, 0])
    
    r_Pm1 = r_P0 - r_m10
    r_Pm2 = r_P0 - r_m20

    rp1 = np.linalg.norm(r_Pm1)
    rp2 = np.linalg.norm(r_Pm2)

    dxdx = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*(x + m2)**2/rp1**5 + 3*m2*(x - 1 + m2)**2/rp2**5
    dxdy = 3*m1*(x + m2)*y/rp1**5 + 3*m2*(x - 1 + m2)*y/rp2**5
    dxdz = 3*m1*(x + m2)*z/rp1**5 + 3*m2*(x - 1 + m2)*z/rp2**5
    dydy = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*y**2/rp1**5 + 3*m2*y**2/rp2**5
    dydz = 3*m1*y*z/rp1**5 + 3*m2*y*z/rp2**5
    dzdz = 1 - m1/rp1**3 - m2/rp2**3 + 3*m1*z**2/rp1**5 + 3*m2*z**2/rp2**5

    Z = np.zeros([3,3])
    I = np.identity(3)
    A = np.array([[dxdx, dxdy, dxdz],
            [dxdy, dydy, dydz],
            [dxdz, dydz, dzdz]])
    Phi = np.block([[Z, I],
           [A, Z]])
    return Phi

def calcdFx(freeVar,mu_star,m1,m2):
    """Calculates the jacobian of the free variables wrt the constraints

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    
    s_T = stateProp(freeVar,mu_star)
    state = s_T[-1]
    
    Phi = calcMonodromyMatrix(state, mu_star,m1,m2)

    phis = np.array([[Phi[1,0], Phi[1,2], Phi[1,4]],
                    [Phi[3,0], Phi[3,2], Phi[3,4]],
                    [Phi[5,0], Phi[5,2], Phi[5,4]]])

    X = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    dw = CRTBP_EOM(freeVar[-1], X, mu_star)
    ddT = np.array([dw[1], dw[3], dw[5]])
    dFx = np.zeros([3,4])
    dFx[:,0:3] = phis
    dFx[:,3] = ddT
    return dFx

def calcFx(freeVar):
    """Applies constraints to the free variables

    Args:
        freeVar (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    s_T = stateProp(freeVar,mu_star)
    state = s_T[-1]

    Fx = np.array([state[1], state[3], state[5]])
    return Fx

def stateProp(freeVar,mu_star):
    """Propagates the dynamics using the free variables

    Args:
        state (float):
            Position in non dimensional units
        eom (float):
            Non-dimensional mass parameter


    Returns:
        ~numpy.ndarray(float):
            jacobian of the free variables wrt the constraints

    """
    x0 = [freeVar[0], 0, freeVar[1], 0, freeVar[2], 0]
    T = freeVar[3]

    sol_int = solve_ivp(CRTBP_EOM, [0, T], x0, args=(mu_star,),rtol=1E-12,atol=1E-12,t_eval=np.linspace(0,T,300))
    states = sol_int.y.T
    
    return states
    
def fsolve_eqns(w,z,solp):
    Fx = calcFx(w)
    zeq = z.T@(w-solp)
    sys_w = np.append(Fx,zeq)

    return sys_w

mu_star = 1.215059*10**(-2)
m1 = (1 - mu_star)
m2 = mu_star

IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 0.681604840704215]
X = [IC[0], IC[2], IC[4], IC[6]]

eps = 3E-6
N = 5
solutions = np.zeros([N,4])
zT = np.array([0, 0, 0, 1])
z = np.array([0, 0, 0, 1])
step = 1E-2

error = 10
ctr = 0
while error > eps:
    Fx = calcFx(X)

    error_new = np.linalg.norm(Fx)

    if error_new > error:
        print('Solution Did Not Converge')
        print(error_new)
        break
    else:
        error = error_new
        ctr = ctr + 1

    dFx = calcdFx(X,mu_star,m1,m2)

    X = X - dFx.T@(np.linalg.inv(dFx@dFx.T)@Fx)

IV = np.array([X[0], X[1], X[2], 2*X[3]])
solutions[ii] = IV
states = stateProp(IV,mu_star)

ax.plot(states[:,0],states[:,1],states[:,2])

## full force model here

    
plt.show()
breakpoint()
