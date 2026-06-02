import scipy.io
import numpy as np
import numpy.linalg as la
from sympy import *
import os
import os.path
from scipy.linalg import lu_factor, lu_solve, eigh
from STMint import STMint
import matplotlib
import matplotlib.pyplot as plt
import pdb
import dill #use dill to save lambda-fied class
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from shapely.geometry import Point, Polygon
import itertools
from scipy.interpolate import CubicSpline

def projection(STM_full,STT_full,state_full,J_max,dim1,dim2):
    # This function takes STMs and STTs for a particular initial condition and makes a
    # projection plot in dim1,dim2 space of the hyperellipsoids for each sampled point.

    # INPUTS: 
    # STM_full is the set of state transition matrices starting from t = 0 to t = end time
    # STT_full is the set of state transition tensors starting from t = 0 to t = end time
    # state_full is the set of states starting from t = 0 to t = end time
    # J_max is the max energy cost associated with the orbit
    # dim1 and dim2 are strings describing the plot dimensions on the x-axis and y-axis, respectively
        # dim1 MUST be either 'X', 'Y', 'Z', 'Xdot', 'Ydot', and 'Zdot'. Same goes for dim2. 
    
    # OUTPUT:
    # ellipse_info is an array of size [rows of state_full x 5] containing info for ellipses. Columns (in order) are :
        # dim1-coordinate center (ex: x-coordinate center, xdot coordinate center, zdot coordinate center)
        # dim2-coordinate center
        # width (so 2 times the semi-major axis)
        # height (so 2 times the semi-minor axis)
        # rotation angle (radians) from x-axis to semi-major axis. Bounded between 0 and pi.

    fig, ax = plt.subplots()
    ax = plt.gca()
    # plt.title('Hyperellipsoid Projections in 2D Plane')
    ellipses = []

    # establish projection matrix and relevant indices for plotting based on dim1 and dim2
    A = np.zeros((2,6)) 
    if dim1 == 'X':
        s1 = 0
    elif dim1 == 'Y':
        s1 = 1
    elif dim1 == 'Z':
        s1 = 2
    elif dim1 == 'Xdot':
        s1 = 3
    elif dim1 == 'Ydot':
        s1 = 4
    elif dim1 == 'Zdot':
        s1 = 5

    if dim2 == 'X':
        s2 = 0
    elif dim2 == 'Y':
        s2 = 1
    elif dim2 == 'Z':
        s2 = 2
    elif dim2 == 'Xdot':
        s2 = 3
    elif dim2 == 'Ydot':
        s2 = 4
    elif dim2 == 'Zdot':
        s2 = 5

    # establish axes labels based on specified projection space
    if dim1 == 'X' or dim1 == 'Y' or dim1 == 'Z':
        plt.xlabel(dim1 + " [DU]")
    else:
        if dim1 == 'Xdot':
            plt.xlabel(r"$\overset{\bullet}{X}$ [DU/TU]")
        elif dim1 == 'Ydot':
            plt.xlabel(r"$\overset{\bullet}{Y}$ [DU/TU]")
        else:
            plt.xlabel(r"$\overset{\bullet}{Z}$ [DU/TU]")

    if dim2 == 'X' or dim2 == 'Y' or dim2 == 'Z':
        plt.ylabel(dim2 + " [DU]")
    else:
        if dim2 == 'Xdot':
            plt.ylabel(r"$\overset{\bullet}{X}$ [DU/TU]")
        elif dim2 == 'Ydot':
            plt.ylabel(r"$\overset{\bullet}{Y}$ [DU/TU]")
        else:
            plt.ylabel(r"$\overset{\bullet}{Z}$ [DU/TU]")

    # place 1's in relevant positions of projection matrix A:
    A[0,s1] = 1
    A[1,s2] = 1

    ref_trajectory, = plt.plot(state_full[:,s1], state_full[:,s2], color=[0,0,0], label='Reference Trajectory') # plot the reference trajectory in black

    [rows,columns,depth] = STM_full.shape # dimensions of STM_full
    ellipse_info = np.zeros((rows,5)) # initialize ellipse_info matrix

    for i in range(0,rows):

        STM = STM_full[i,:,:] @ STM_full[-1,:,:] @ la.inv(STM_full[i,:,:]) # STM for current timestep
        STT = STT_full[i,:,:,:] # STT for current timestep

        Matrix1 = np.block([[np.identity(6), np.zeros((6,6))],
        [-la.solve(STM[:6, 6:12], STM[:6, :6]), la.inv(STM[:6, 6:12])]]) # construct Matrix1 from STMS

        # work with 13 x  13 matrices:
        # TempMatrix1 = la.inv(np.transpose(STM_full[i,:,:])) @ (STT_full[-1,12,:,:] - STT_full[i,12,:,:]) @ la.inv(STM_full[i,:,:])
        # TempMatrix2 = np.transpose(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))@STT_full[i,12,:,:]@(STM_full[-1,:,:]@la.inv(STM_full[i,:,:]))+TempMatrix1

        # work with 12 x 12 matrices:
        TempMatrix1 = la.inv(np.transpose(STM_full[i,:12,:12])) @ (STT_full[-1,12,:12,:12] - STT_full[i,12,:12,:12]) @ la.inv(STM_full[i,:12,:12])
        TempMatrix2 = np.transpose(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))@STT_full[i,12,:12,:12]@(STM_full[-1,:12,:12]@la.inv(STM_full[i,:12,:12]))+TempMatrix1
        Matrix2 = TempMatrix2[:12,:12]

        # eigh returns eigenvalues in ascending order with eigenvectors in their corresponding positions
        E = np.transpose(Matrix1) @ Matrix2 @ Matrix1  # establish E-matrix
        E_star = np.block([[np.identity(6), np.identity(6)]]) @ E @ np.transpose(np.block([[np.identity(6), np.identity(6)]])) # establish Estar matrix
        gamma, w = eigh(E_star) # get eigenvalues and eigenvectors of E_star
        
        # Adjustments to compensate for degeneracy:
        w_adj = w[:,1:6] # remove first eigenvector (the one with infinite extent)
        gamma_adj = gamma[1:6] # remove first eigenvalue (the one with infinite extent)
        Qprime = np.transpose(w_adj) # establish Q matrix with row-vectors corresponding to eigenvectors 
        Eprime = np.diag(gamma_adj) # form adjusted eigenvalue matrix

        lam, z = eigh(Qprime @ np.transpose(A) @ A @ np.transpose(Qprime), Eprime) # solve adjusted problem for new eigenvalues and eigenvectors
        
        # normalize eigenvectors
        z1 = z[:,3] / la.norm(z[:,3]) 
        z2 = z[:,4] / la.norm(z[:,4])

        # get directions of projected energy ellipsoid
        dir1 = A @ (np.transpose(Qprime)) @ z1 
        dir2 = A @ (np.transpose(Qprime)) @ z2
        angle = np.arctan2(dir1[1],dir1[0]) # radians, angle of ellipse rotation


        ext1 = 2*np.sqrt(2*J_max*lam[3]) #  width of ellipse projection
        ext2 = 2*np.sqrt(2*J_max*lam[4]) # height of ellipse projection

        rotation = angle*180/np.pi

        position = np.array([state_full[i,s1],state_full[i,s2]])
        
        # ellipse plotting:
        ellipse = matplotlib.patches.Ellipse(xy=(position[0],position[1]), width=ext1, height=ext2, edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=rotation)

        ax.add_patch(ellipse)
        ellipses.append(ellipse)
            

        # ellipse_info is n x 5 array storing the x-coordinate of center, y-coordinate of center, semi-major axis, semi-minor axis, and rotation angle
        ellipse_info[i,0] = position[0]
        ellipse_info[i,1] = position[1]

        # Ensure larger extent is included in column 2 of ellipse_info:
        if ext1 > ext2: # extent 1 is larger extent
            ellipse_info[i,2] = ext1
            ellipse_info[i,3] = ext2
            ellipse_info[i,4] = np.arctan2(dir1[1],dir1[0]) # rad, rotation angle from x-axis to semi-major axis
        else: # extent 2 is larger extent
            ellipse_info[i,2] = ext2
            ellipse_info[i,3] = ext1
            ellipse_info[i,4] = np.arctan2(dir2[1],dir2[0]) # rad, rotation angle from x-axis so semi-major axis

        if ellipse_info[i,4] < 0:
            ellipse_info[i,4] = ellipse_info[i,4] + np.pi # correction so that ellipse rotation angle is between 0 and pi
            
        
    plt.legend(["Reference","Reachable Bounds"], loc="upper right") # create legend

    plt.savefig('sampled_projection.pdf', format='pdf')
    plt.show()


    return ellipse_info




def tangent_lines(ellipse_info,limits,first_axis_label,second_axis_label):
    # This function utilizes scipy.optimize to determine solutions for the tangent-line problem between two rotated ellipses
    # for the ellipses in ellipse_info.
    # Every time the word "problem" is used in comments, it refers to the problem of finding the two tangent lines for a set of two ellipses.
    # "Version" refers to the problem-type being attempted, which will either be the unrotated version or rotated version
        # Rotating the problem prevents large-sloping tangent lines from complicating the nonlinear solver, increasing the chances of solutions being found.
        # This entails rotating the characteristics of the two ellipses in a problem such that the line connecting their centers has a certain slope (which you can specify). The best slope was 0 by inspection. 

    # INPUTS:
    # ellipse_info is a [rows x 5] numpy array of ellipses. Columns (in order contain):
        # first-axis coordinate center
        # second-axis coordinate center
        # 2*semi-major axis
        # 2*semi-minor axis
        # rotation angle (rad) from x-axis to semi-major axis
    # limits is a list describing the first and last index of ellipse_info you'd like to create sets of tangent lines for. It should look like [first_index,second_index].
        # Ex: For an ellipse_info array of size 260x5, limits is [0,259] if you want to solve every problem in the array. 
    # first_axis_label is a string describing the x-axis label on your outputted plot
    # second_axis_labek is a string describing the y-axis label on your outputted plot

    def cot(theta): # basic cotangent function created
        return 1 / np.tan(theta) 

    needed_solutions = (limits[1] - limits[0])*2
    total_solutions = 0
    pair_found = False

    # determine max and min coordinates (and max semi-major axis) from set of ellipses to establish graph's axis limits:
    hmax = np.max(ellipse_info[:,0])
    hmin = np.min(ellipse_info[:,0])
    kmax = np.max(ellipse_info[:,1])
    kmin = np.min(ellipse_info[:,1])
    amax = np.max(ellipse_info[:,2])

    # set up graph parameters
    fig, ax = plt.subplots()
    ax = plt.gca()
    ax.set_xlim([hmin-amax, hmax+amax]) 
    ax.set_ylim([kmin-amax, kmax+amax])
    ax.set_aspect('equal', adjustable='box') 

    plt.plot(ellipse_info[:,0],ellipse_info[:,1],linewidth=1.5,color='k',label = 'Reference')

    solution1_array = np.zeros((limits[1]-limits[0],7))
    solution2_array = np.zeros((limits[1]-limits[0],7))

    C = 1000 # scaling factor for ellipses (center-coordinates and axes lengths are multiplied by this number and scaled back down at end)
    pi = np.pi
    rotation_angle = 0 # rad, slope of line between ellipses' centers if attempting rotated problem
    last_successful = "unrotated" # last version of the problem that was successful (either "unrotated" or "rotated"). Initialized as unrotated.
    

    for i in range(limits[0],limits[1]):
        
        solutions_found = 0 # no solutions found initially
        k = 0
        runthroughs = 0 # number of attempts at current problem. If current problem (either rotated or unrotated problem) fails, runthroughs increases to 1 and either rotated or unrotated prob is attempted
        print(" ")
        print("Current iteration: " + str(i))

        ell1 = ellipse_info[i,:] # characteristics of current ellipse
        ell2 = ellipse_info[i+1,:] # characteristics of next ellipse

        # ellipse 1 constants (with relevant scaling)
        h1 = ell1[0] * C # x-coordinate center
        k1 = ell1[1] * C # y-coordinate center
        a1 = ell1[2]/2 * C # semi-major axis
        b1 = ell1[3]/2 * C # semi-minor axis
        r1 = ell1[4] # rotation angle to semi-major axis (rad)

        # ellipse 2 constants (same characteristics described above but for second ellipse)
        h2 = ell2[0] * C
        k2 = ell2[1] * C
        a2 = ell2[2]/2 * C
        b2 = ell2[3]/2 * C
        r2 = ell2[4]

        # plot ellipse
        ellipse = matplotlib.patches.Ellipse(xy=(ell1[0],ell1[1]), width=ell1[2], height=ell1[3], edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=ell1[4]*180/np.pi)
        ax.add_patch(ellipse)


        while (solutions_found < 2) and (k < 16) and (runthroughs < 2):
            # solutions_found is the number of solutions found for the current problem. Every problem has two solutions, so if two are found the while loop is exited.
            # k is the index of the attempted interval (explained later). All possible intervals are attempted before moving on to rotated or unrotated version of problem.
            # runthroughs is described earlier. This function attempts whatever version of the problem was most recently successful (unrotated v. rotated) first,
            # and then moves onto other version if it fails. This is meant to speed up the code.

            if (last_successful == "rotated" and runthroughs == 0) or (last_successful == "unrotated" and runthroughs > 0):
                # Rotated version of the prob is being attempted, either because rotation was successful in previous problem
                # or unrotated version failed on the initial attempt of the current problem.

                m0 = (k2 - k1) / (h2 - h1) # slope of line connecting ellipses' centers
                slope_angle = np.arctan2(m0,1) # radians, angle of line connecting ellipses

                supp_angle = rotation_angle - slope_angle # rotation angle that brings slope line to rotation_angle

                # rotate center coordinates and rotation angles of ellipses
                h1_rot = np.cos(supp_angle)*h1 - np.sin(supp_angle)*k1
                k1_rot = np.sin(supp_angle)*h1 + np.cos(supp_angle)*k1
                r1_rot = r1 + supp_angle

                h2_rot = np.cos(supp_angle)*h2 - np.sin(supp_angle)*k2
                k2_rot = np.sin(supp_angle)*h2 + np.cos(supp_angle)*k2
                r2_rot = r2 + supp_angle

                # redefine centers and rotation angles of ellipses
                h1 = h1_rot
                k1 = k1_rot
                r1 = r1_rot

                h2 = h2_rot
                k2 = k2_rot
                r2 = r2_rot


            elif (last_successful == "rotated" and runthroughs > 0):
                # Rotation attempt at problem failed. Use normal characteristics instead of rotated characteristics. 
                # ellipse 1 constants
                h1 = ell1[0] * C
                k1 = ell1[1] * C
                a1 = ell1[2]/2 * C
                b1 = ell1[3]/2 * C
                r1 = ell1[4]

                # ellipse 2 constants
                h2 = ell2[0] * C
                k2 = ell2[1] * C
                a2 = ell2[2]/2 * C
                b2 = ell2[3]/2 * C
                r2 = ell2[4]

            # Another possibility besides the conditions in the if-elif statement above is last_successful == "unrotated" and runthroughts == 0,
            # but ellipse characteristics don't need to be modified if this is the case because it just takes info directly from ellipse_info
            # to attempt an unrotated problem. 


            # define constraints for tangent line connecting two ellipses:
            con1 = lambda x: (h1 + a1*np.cos(x[0])*np.cos(r1) - b1*np.sin(x[0])*np.sin(r1)) - x[2]
            con2 = lambda x: (k1 + a1*np.cos(x[0])*np.sin(r1) + b1*np.sin(x[0])*np.cos(r1)) - x[4]
            con3 = lambda x: (h2 + a2*np.cos(x[1])*np.cos(r2) - b2*np.sin(x[1])*np.sin(r2)) - x[3]
            con4 = lambda x: (k2 + a2*np.cos(x[1])*np.sin(r2) + b2*np.sin(x[1])*np.cos(r2)) - x[5]
            con5 = lambda x: x[6]*(x[3]-x[2]) - (x[5]-x[4])
            con6 = lambda x: ((-b1/a1)*(cot(x[0])) + np.tan(r1)) / (1 + (b1/a1)*(cot(x[0]))*np.tan(r1)) - x[6]
            con7 = lambda x: ((-b2/a2)*(cot(x[1])) + np.tan(r2)) / (1 + (b2/a2)*(cot(x[1]))*np.tan(r2)) - x[6]

            nonlinear_constraints = ({'type': 'eq', 'fun': con1},
                {'type': 'eq', 'fun': con2},
                {'type': 'eq', 'fun': con3},
                {'type': 'eq', 'fun': con4},
                {'type': 'eq', 'fun': con5},
                {'type': 'eq', 'fun': con6},
                {'type': 'eq', 'fun': con7},)
            
            def objective_fun(x):
                # Can use trivial objective (return 0) or quadratic objective (results did not differ much as long as constraints between
                # objectives are also used).  
                eq1 = (h1 + a1*np.cos(x[0])*np.cos(r1) - b1*np.sin(x[0])*np.sin(r1)) - x[2]
                eq2 = (k1 + a1*np.cos(x[0])*np.sin(r1) + b1*np.sin(x[0])*np.cos(r1)) - x[4]
                eq3 = (h2 + a2*np.cos(x[1])*np.cos(r2) - b2*np.sin(x[1])*np.sin(r2)) - x[3]
                eq4 = (k2 + a2*np.cos(x[1])*np.sin(r2) + b2*np.sin(x[1])*np.cos(r2)) - x[5]
                eq5 = x[6]*(x[3]-x[2]) - (x[5]-x[4])
                eq6 = ((-b1/a1)*(cot(x[0])) + np.tan(r1)) / (1 + (b1/a1)*(cot(x[0]))*np.tan(r1)) - x[6]
                eq7 = ((-b2/a2)*(cot(x[1])) + np.tan(r2)) / (1 + (b2/a2)*(cot(x[1]))*np.tan(r2)) - x[6]

                # return eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2 + eq7**2
                return 0
            
            
            two_solutions_found = False # boolean describing if 2 solutions are found
            one_solution_found = False # boolean describing if 2 solutions are found
            # two_solutions_found and one_solution_found are initialized as 0 because current version of the problem has not been attempted
            jump = 1

            # The code below identifies angles for which the tangent-line problem is undefined.
            # This was done to create intervals over which the problem is attempted. Intervals are:
            # (0,phi1),(phi1,pi),(pi,phi2),(phi2,2*pi)
            # Each ellipse in the problem has its own intervals (so each ellipse has its own phi1 and phi2).
            # These intervals are used to establish bounds on the problem. This resulted in more consistent identification
            # of solutions for each problem, likely due to the fact that establishing bounds constrains the search-space.


            # Ellipse 1 phi calculations:
            phi_ell1 = np.arctan(-b1/a1*np.tan(r1))

            if phi_ell1 < 0:
                phi1_ell1 = phi_ell1 + np.pi # make phi1_ell1 between 0 and pi
            else:
                phi1_ell1 = phi_ell1

            phi2_ell1 = phi1_ell1 + np.pi # make phi2_ell1 between pi and 2pi

            # Ellipse 2 phi calculations
            phi_ell2 = np.arctan(-b2/a2*np.tan(r2))
            if phi_ell2 < 0:
                phi1_ell2 = phi_ell2 + np.pi # make phi1_ell1 between 0 and pi
            else:
                phi1_ell2 = phi_ell2

            phi2_ell2 = phi1_ell2 + np.pi # make phi2_ell1 between pi and 2pi

            # Possible intervals for ellipse 1:
            A = [
                (0, phi1_ell1),
                (phi1_ell1, np.pi),
                (np.pi, phi2_ell1),
                (phi2_ell1, 2 * np.pi)
            ]

            # Possible intervals for ellipse 2:
            B = [
                (0, phi1_ell2),
                (phi1_ell2, np.pi),
                (np.pi, phi2_ell2),
                (phi2_ell2, 2 * np.pi)
            ]

            # Generate all combinations of one interval from A and one from B
            combinations = list(itertools.product(A, B)) # combinations are always generated in specific order

            # Rearrange combinations list so more likely combinations are attempted first. Rearrangement was determined by inspection. 
            # This rearrangement is non-essential and is only meant to speed up the code:
            int2 = combinations[15]
            int_replace = combinations[1]

            combinations[1] = int2
            combinations[15] = int_replace

            int2 = combinations[5]
            int_replace = combinations[2]

            combinations[2] = int2
            combinations[5] = int_replace

            int2 = combinations[10]
            int_replace = combinations[1]

            combinations[1] = int2
            combinations[10] = int_replace

            int2 = combinations[10]
            int_replace = combinations[3]

            combinations[3] = int2
            combinations[10] = int_replace
             # End of rearrangement.
            
            solutions_found = 0 # number of solutions found in current version of the problem (initialized at 0 because version has not been attempted).

            d = 0 # index for attempting the most recent successful combinations of intervals

            # Note on the variable d: The code will attempt the successful boundary combinations of the most recent problem first. 
            # For example, if the last problem was first successful at the combo ((0,phi1_ell1),(0,phi1_ell2)), it will attempt this 
            # combo first at d = 0 index. Then if the second successful was ((phi1_ell1,pi),(phi1_ell2,pi)), it will attempt 
            # this combo second at the d = 1 index. If it doesn't find two solutions, it will just move on to the list of bound
            # combinations until finding 2 solutions. The combinations at d = 0 and d = 1 are most likely to succeed because the 
            # previous problem succeeded at these combinations, so it is useful to attempt these first to speed up the code. 

            while (solutions_found < 2) and (k < len(combinations)):

                # A solution consists of the following variables:
                # p is the angle relative to the ellipse's major axis at which the tangent line collides with the ellipse.
                # However, this angle actually corresponds to the angle if the ellipse were a circle (similar to eccentric anomaly).
                # p1 and p2 are the angles p (in radians) for ellipse 1 and ellipse 2, respectively.
                # (x1,y1) are coordinates of tangent line's collision with the first ellipse.
                # (x2,y2) are coordinates o fthe tangent line's collision with the second ellipse.
                # m is the slope of the tangent line.

                # Initialize guesses for variables:
                # Initialize p1 and p2 at center of bounds:
                p10 = (combinations[k][0][0] + combinations[k][0][1]) / 2 
                p20 = (combinations[k][1][0] + combinations[k][1][1]) / 2
                # Initialize (x1,y1) and (x2,y2) at points on ellipses 1 and 2 corresponding to p1 and p2, respectively:
                x10 = h1 + a1*np.cos(r1)*np.cos(p10) - b1*np.sin(r1)*np.sin(p10)
                y10 = k1 + a1*np.sin(r1)*np.cos(p10) + b1*np.cos(r1)*np.sin(p10)
                x20 = h2 + a2*np.cos(r2)*np.cos(p20) - b2*np.sin(r2)*np.sin(p20)
                y20 = k2 + a2*np.sin(r2)*np.cos(p20) + b2*np.cos(r2)*np.sin(p20)

                m0 = (k2 - k1) / (h2 - h1) # initialize slope of tangent line as line connecting ellipses' centers
                # m0 = (y20 - y10) / (x20- x10)

                # Select bounds based current iteration of combinations:
                p1bnds = combinations[k][0]
                p2bnds = combinations[k][1]

                p1low = combinations[k][0][0]
                p1high = combinations[k][0][1]

                p2low = combinations[k][1][0]
                p2high = combinations[k][1][1]


                if d == 0 and i > limits[0] and pair_found == True:
                    # Override initial guesses and bounds if on index d = 0 in order to attempt most recent successful bounds first
                    # If pair_found == true, then at least one pair of solutions has been found for at least one problem in the set of problems.
                    # pair_found exists because it is only possible to attempt a most recent combination of bounds if one or more problems have been fully solved. 
                    p10 = (combinations[k_successful1][0][0] + combinations[k_successful1][0][1]) / 2
                    p20 = (combinations[k_successful1][1][0] + combinations[k_successful1][1][1]) / 2
                    x10 = h1 + a1*np.cos(r1)*np.cos(p10) - b1*np.sin(r1)*np.sin(p10)
                    y10 = k1 + a1*np.sin(r1)*np.cos(p10) + b1*np.cos(r1)*np.sin(p10)
                    x20 = h2 + a2*np.cos(r2)*np.cos(p20) - b2*np.sin(r2)*np.sin(p20)
                    y20 = k2 + a2*np.sin(r2)*np.cos(p20) + b2*np.cos(r2)*np.sin(p20)
                    m0 = (k2 - k1) / (h2 - h1)
                    # m0 = (y20 - y10) / (x20 - x10)


                    p1bnds = combinations[k_successful1][0]
                    p2bnds = combinations[k_successful1][1]

                    p1low = combinations[k_successful1][0][0]
                    p1high = combinations[k_successful1][0][1]

                    p2low = combinations[k_successful1][1][0]
                    p2high = combinations[k_successful1][1][1]

                elif d == 1 and i > limits[0] and pair_found == True:
                    # Override initial guesses for d = 1 index. Same logic as comments immediately following if statment. 
                    p10 = (combinations[k_successful2][0][0] + combinations[k_successful2][0][1]) / 2
                    p20 = (combinations[k_successful2][1][0] + combinations[k_successful2][1][1]) / 2
                    x10 = h1 + a1*np.cos(r1)*np.cos(p10) - b1*np.sin(r1)*np.sin(p10)
                    y10 = k1 + a1*np.sin(r1)*np.cos(p10) + b1*np.cos(r1)*np.sin(p10)
                    x20 = h2 + a2*np.cos(r2)*np.cos(p20) - b2*np.sin(r2)*np.sin(p20)
                    y20 = k2 + a2*np.sin(r2)*np.cos(p20) + b2*np.cos(r2)*np.sin(p20)
                    # m0 = (k2 - k1) / (h2 - h1)
                    m0 = (y20 - y10) / (x20 - x10)


                    p1bnds = combinations[k_successful2][0]
                    p2bnds = combinations[k_successful2][1]

                    p1low = combinations[k_successful2][0][0]
                    p1high = combinations[k_successful2][0][1]

                    p2low = combinations[k_successful2][1][0]
                    p2high = combinations[k_successful2][1][1]

                # Establish x and y bounds based on length of major axes:
                x1bnds = (h1-a1,h1+a1)
                x2bnds = (h2-a2,h2+a2)
                y1bnds = (k1-a1,k1+a1)
                y2bnds = (k2-a2,k2+a2)

                mbnds = (None,None)

                bnds = (p1bnds,p2bnds,x1bnds,x2bnds,y1bnds,y2bnds,mbnds) # fill in bounds

                X0 = [p10, p20, x10, x20, y10, y20, m0] # fill in initial guess
                solution = minimize(objective_fun,X0, method='SLSQP',bounds = bnds, tol=1e-10,constraints=nonlinear_constraints,options={'maxiter': 50}) # attempt nonlinear system

                # solution = minimize(objective_fun,X0, method='SLSQP',bounds = bnds, tol=1e-10,options={'maxiter': 50})

                # Check that tangent line does not cross line connecting ellipses' centers:
                m_ell = (k2 - k1) / (h2 - h1) # slope connecting ellipse centers
                x1 = solution.x[2]
                x2 = solution.x[3]
                y1 = solution.x[4]
                m_tan = solution.x[6] # slope of the tangent line
                x_cross = ((y1 - k1) + (m_ell*h1 - m_tan*x1)) / (m_ell - m_tan)

                if x1 > x2:
                    if (x_cross > x2) and (x_cross < x1):
                        solution.success = False  # throw solution away if lines cross
                else:
                    if (x_cross > x1) and (x_cross < x2):
                        solution.success = False # throw solution away if lines cross


                if solutions_found > 0 and solution.success == True:  
                    check = np.isclose(solution.x,solution1) # See if the second solution matches first solution. If yes, throw it away because you want two unique solutions.

                    for b in range(0,len(check)): 
                        if check[b] == False: # second solution is different from first (good)
                            matching = 0 
                        else: # second solution is the same as the first (bad)
                            matching = 1
                    
                    if matching == 1:
                        solution.success = False # throw current solution away if it matches the first

                if solution.success == True:
                    solutions_found = solutions_found + 1 # add to number of unique solutions found for this version of the problem
                    solution = solution.x # solution data

                    if solutions_found == 1:
                        solution1 = solution # identify solution as solution1 if it is the first solution found
                        if d > 1:
                            k_successful1 = k # reassign k_successful1 if you enter list of combos phase, which only happens if first two attempts fail (d>1)
                        
                    elif solutions_found == 2:
                        solution2 = solution # identify solution as solution2 if it is the second solution found

                        if d > 1: 
                            k_successful2 = k # reassign k_successful2 if you enter list of combos phase, which only happens if first two attempts fail (d>

                if d == 1 and solutions_found < 2: 
                    solutions_found = 0 # say no solutions are found if two solutions aren't found in the initial two attempts. Move to list phase.
                    
                if d < 2: # If d < 2, you are in initial attempt phase
                    d = d + 1 # Add only to index d
                else: # If d>= 2, you are in list of combos phase
                    k = k + 1 # Add only to index k


            if solutions_found == 0: # no solutions found, just say both are most recent solution
                solution1 = solution.x
                solution2 = solution1

            elif solutions_found == 1: # one solution found, just say second solution is same as first
                solution2 = solution1

            else: # two solutions found
                pair_found = True
                if runthroughs > 0: # attempt at original problem failed
                    print("Last successful")
                    if last_successful == "unrotated": # unrotated problem failed, rotated problem succeeded
                        last_successful = "rotated" # switch last successful to rotated
                    else:
                        last_successful = "unrotated" # switch last successful to unrotated




            # if runthroughs > 0:
            if last_successful == "rotated": # most recent success was a rotated problem, so rotate solution back
                #  ROTATE SOLUTION BACK
                x1rot1 = np.cos(supp_angle)*solution1[2] + np.sin(supp_angle)*solution1[4]
                y1rot1 = -np.sin(supp_angle)*solution1[2] + np.cos(supp_angle)*solution1[4]

                x1rot2 = np.cos(supp_angle)*solution1[3] + np.sin(supp_angle)*solution1[5]
                y1rot2 = -np.sin(supp_angle)*solution1[3] + np.cos(supp_angle)*solution1[5]


                x2rot1 = np.cos(supp_angle)*solution2[2] + np.sin(supp_angle)*solution2[4]
                y2rot1 = -np.sin(supp_angle)*solution2[2] + np.cos(supp_angle)*solution2[4]

                x2rot2 = np.cos(supp_angle)*solution2[3] + np.sin(supp_angle)*solution2[5]
                y2rot2 = -np.sin(supp_angle)*solution2[3] + np.cos(supp_angle)*solution2[5]

                solution1[2] = x1rot1
                solution1[3] = x1rot2
                solution1[4] = y1rot1
                solution1[5] = y1rot2
                solution1[6] = (solution1[5] - solution1[4]) / (solution1[3] - solution1[2])

                solution2[2] = x2rot1
                solution2[3] = x2rot2
                solution2[4] = y2rot1
                solution2[5] = y2rot2
                solution2[6] = (solution2[5] - solution2[4]) / (solution2[3] - solution2[2])


            # added something here
            for j in range(2,7):
                solution1[j] = solution1[j] / C
                solution2[j] = solution2[j] / C

            solution1_array[i-limits[0],:] = solution1
            solution2_array[i-limits[0],:] = solution2

            if runthroughs == 0:
                print("Runthrough 1 results")
                if solutions_found == 0:
                    print("NO SOLUTIONS FOUND FOR THIS ITERATION")
                elif solutions_found == 1:
                    print("ONE SOLUTION FOUND FOR THIS ITERATION")
                    print(solution1)
                else:
                    print("TWO SOLUTIONS FOUND FOR THIS ITERATION")
            else:
                print("Runthrough 2 results")
                if solutions_found == 0:
                    print("NO SOLUTIONS FOUND FOR THIS ITERATION")
                elif solutions_found == 1:
                    print("ONE SOLUTION FOUND FOR THIS ITERATION")
                    print(solution1)
                else:
                    print("TWO SOLUTIONS FOUND FOR THIS ITERATION")

                    
            if solutions_found < 2:
                solutions_found = 0 # reset solutions_found to 0 if 2 aren't found for next runthrough
                k = 0

            runthroughs = runthroughs + 1


        total_solutions = total_solutions + solutions_found



        # plot solution lines
        x_points1 = np.array([solution1[2],solution1[3]])
        y_points1 = np.array([solution1[4],solution1[5]])
        plt.plot(x_points1,y_points1,linewidth= 1.5,color = 'b')

        x_points2 = np.array([solution2[2],solution2[3]])
        y_points2 = np.array([solution2[4],solution2[5]])
        if i == limits[0]:
            plt.plot(x_points2,y_points2,linewidth= 1.5,color = 'b',label='Reachable Set Boundary')
        else:
            plt.plot(x_points2,y_points2,linewidth= 1.5,color = 'b')

    if first_axis_label == 'X' or first_axis_label == 'Y' or first_axis_label == 'Z':
        plt.xlabel(first_axis_label + " [DU]")
    else:
        if first_axis_label == 'Xdot':
            plt.xlabel(r"$\overset{\bullet}{X}$ [DU/TU]")
        elif first_axis_label == 'Ydot':
            plt.xlabel(r"$\overset{\bullet}{Y}$ [DU/TU]")
        else:
            plt.xlabel(r"$\overset{\bullet}{Z}$ [DU/TU]")

    if second_axis_label == 'X' or second_axis_label == 'Y' or second_axis_label == 'Z':
        plt.ylabel(second_axis_label + " [DU]")
    else:
        if second_axis_label == 'Xdot':
            plt.ylabel(r"$\overset{\bullet}{X}$ [DU/TU]")
        elif first_axis_label == 'Ydot':
            plt.ylabel(r"$\overset{\bullet}{Y}$ [DU/TU]")
        else:
            plt.ylabel(r"$\overset{\bullet}{Z}$ [DU/TU]")

    plt.legend(loc="upper right")
    # plt.title("Projections with Tangent Lines")

    if total_solutions < needed_solutions:
        print("MISSING SOLUTIONS, " + str(total_solutions) + " solutions found, " + str(needed_solutions) + " needed.")
    else:
        print("ALL SOLUTIONS FOUND.")

    plt.savefig(first_axis_label + second_axis_label + '_tangent_lines.pdf', format='pdf')
    plt.show()


        
    return solution1_array,solution2_array


def interpolate_ellipses(ellipse_info,p1,p2):
    # This function takes an array containing information describing ellipses and creates an array of interpolated ellipses using CubicSpline.
    # It will also make a plot containing the reference trajectory and the projections. 
        # Ideally, enough interpolation should be performed so that individual ellipses
        # are indistinguishable from each other in the unzoomed figure that is produced. 
        # It should look like 1 continuous blob around the reference trajectory. 

    # ellipse_info is a rows x 5 numpy array containing ellipse characteristics
        # First column contains p1-coordinates of ellipses' centers
        # Second column contains p2-coordinates of ellipses' centers
        # Third column contains the first extent of the ellipse
        # Second column contains the second extent of the ellipse
        # Fifth column contains the rotation angle (in radians) of the ellipse
    # p1 is a string describing the x-axis of the projection space. MUST BE 'X','Y','Z', 'Xdot','Ydot', or 'Zdot'. (Upper-case matters)
    # p2 is a string describing the y-axis of the projection space. MUST BE 'X','Y','Z', 'Xdot','Ydot', or 'Zdot'. (Upper-case matters)

    # ellipse_info_ip is the output of the function and is an n x 5 array describing the characteristics of the interpolated ellipses
        # Columns of ellipse_info_ip contain characteristics in same order as ellipse_info

    [rows,columns] = ellipse_info.shape # dimensions of ellipse_info
    n = 30*rows # number of samples in interpolation vector
    x = np.linspace(0,rows,num = rows) # vector of sampled points in original array
    xs = np.linspace(0,rows, num = n) # vector of sampled points in interpolated array (len(xs) > len(x))

    h = ellipse_info[:,0] # p1-coordinate centers of ellipses
    k = ellipse_info[:,1] # p2-coordinate centers of ellipses
    ext1 = ellipse_info[:,2] # first extent of ellipses
    ext2 = ellipse_info[:,3] # second extent of ellipses
    r = ellipse_info[:,4] # rotation angle of ellipses

    # Unwrapping-angles procedure (prevents huge drops or rises in angle in r vector):
    r_new = np.zeros((rows)) # create empty vector
    r_new[0] = r[0] # initialize r_new[0]

    for i in range(1,rows):
        difference = r[i] - r_new[i-1] # difference between current angle and previous angle

        if difference > np.pi / 2: # check if difference is very large
            r_new[i] = r[i] - np.pi # unwrap angle
        elif difference < -np.pi / 2: # check if difference is very large (but negative this time)
            r_new[i] = r[i] + np.pi # unwrap angle
        else:
            r_new[i] = r[i] # keep angle if differnence is very small

    r = r_new # reset r to r_new
    # The above procedure is needed to prevent r vector (converted to degrees) from looking like, for example, [...,178,179.5,1,2,...].
    # This would cause problems for interpolation. It would convert above example to [...178,179.5,181,182,...], which is now 
    # interpreted (correctly) as a small angle difference between vector entries.

    # interpolate points using CubicSpline:
    h_ip = CubicSpline(x,h)(xs)
    k_ip = CubicSpline(x,k)(xs)
    ext1_ip = CubicSpline(x,ext1)(xs)
    ext2_ip = CubicSpline(x,ext2)(xs)
    r_ip = CubicSpline(x,r)(xs)

    # fill in information for array of interpated ellipses:
    ellipse_info_ip = np.zeros((n,5))
    ellipse_info_ip[:,0] = h_ip
    ellipse_info_ip[:,1] = k_ip
    ellipse_info_ip[:,2] = ext1_ip
    ellipse_info_ip[:,3] = ext2_ip
    ellipse_info_ip[:,4] = r_ip

    # create plot characteristics
    fig, ax = plt.subplots()
    ax = plt.gca()
    # plt.title('Hyperellipsoid Projections in 2D Plane')

    # determine max and min characteristics of ellipses for purposes of establishing axis limits on figure
    hmax = np.max(ellipse_info[:,0])
    hmin = np.min(ellipse_info[:,0])
    kmax = np.max(ellipse_info[:,1])
    kmin = np.min(ellipse_info[:,1])
    amax = np.max(ellipse_info[:,2])

    # set up axis characteristics. (Uncomment if you want to fix axes and/or set axes equal)
    ax.set_xlim([hmin-1.25*amax, hmax+1.25*amax]) 
    ax.set_ylim([kmin-1.25*amax, kmax+1.25*amax])
    ax.set_aspect('equal', adjustable='box') 

    ref_trajectory, = plt.plot(h, k, color=[0,0,0], label='Reference Trajectory') # reference trajectory points

    # plot interpolated ellispes:
    for i in range(0,n):
        ellipse = matplotlib.patches.Ellipse(xy=(h_ip[i],k_ip[i]), width=ext1_ip[i], height=ext2_ip[i], edgecolor=[252/255, 227/255, 3/255], fc=[252/255, 227/255, 3/255], angle=r_ip[i]*(180/np.pi))

        ax.add_patch(ellipse)

    # Create x-axis label based on p1
    if p1 == 'X' or p1 == 'Y' or p1 == 'Z': # check p1
        plt.xlabel(p1 + " [DU]") # make units DU if p1 describes a position axis
    else:
        if p1 == 'Xdot':
            plt.xlabel(r"$\overset{\bullet}{X}$ [DU/TU]")
        elif p1 == 'Ydot':
            plt.xlabel(r"$\overset{\bullet}{Y}$ [DU/TU]")
        else:
            plt.xlabel(r"$\overset{\bullet}{Z}$ [DU/TU]")

    # Do same process as above for p2 to create y-axis label
    if p2 == 'X' or p2 == 'Y' or p2 == 'Z':
        plt.ylabel(p2 + " [DU]")
    else:
        if p2 == 'Xdot':
            plt.ylabel(r"$\overset{\bullet}{X}$ [DU/TU]")
        elif p2 == 'Ydot':
            plt.ylabel(r"$\overset{\bullet}{Y}$ [DU/TU]")
        else:
            plt.ylabel(r"$\overset{\bullet}{Z}$ [DU/TU]")

    plt.legend(["Reference", "Reachable Bounds"], loc="upper right") # create legend
    plt.savefig(str(p1)+str(p2)+'_projection.pdf', format='pdf')
    plt.show()

    return ellipse_info_ip

# FUNCTIONS
# Detailed function descriptions are present inside functions. A brief summary is shown below:

# projection 
# Takes in array of STMs, STTs, states. It also takes in energy cost J_max. It takes
# dimension 1 and dimension 2 of projection space which must be 'X','Y','Z','Xdot','Ydot','Zdot'
# It outputs ellipse_info, which is an n x 5 array (where n is the number of STMs).
# See the function for description of columns of the array.

# tangent_lines
# Takes in ellipse_info array (output of projection function). Also specifiy axes labels,
# which must follow same convention as dim1 and dim2 from projection function
# outputs solution1_array and solution2_array, which describes solutions to 2*(n-1) tangent line problems
# where n is the number of STMs again.

# interpolate_ellipses
# Takes in ellipse_info array (output of projection function. Specify axes labels in the same manner
# as previous functions.
# Returns interpolated_ellipses array which has characteristics of ellipses determined through CubicSpline interpolation


# function calls are below (must define STM_full,STT_full,J_max)
ellipse_info = projection(STM_full,STT_full,state_full,J_max,'X','Y')
limits = [0,259]
[solution1_array,solution2_array] = tangent_lines(ellipse_info,limits,"X","Y")
ellipse_info_interpolated = interpolate_ellipses(ellipse_info,'X','Y')