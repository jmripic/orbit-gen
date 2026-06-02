# This function is used to test the angular_diameter method included in the orbit_eval class in
# the starlift metrics class. A few calculations were performed on Desmos using different orbits
# at different points in time to check that the method outputs the same values. 
# All tests pass as of 5/17/24.

import sys
sys.path.insert(1, sys.path[0][0:-10])
print(sys.path)
from metricTests import Starlift_metrics
# from tools.Solution import Solution
import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
from scipy import integrate as int
import unittest


# load pickle files for DRO_11 and DRO_13
DRO_11 = Starlift_metrics.load_pickle_file("orbitFiles/DRO_11.241_days.p")
DRO_11 = Starlift_metrics.orbit_eval(DRO_11)

DRO_13 = Starlift_metrics.load_pickle_file("orbitFiles/DRO_13.0486_days.p")
DRO_13 = Starlift_metrics.orbit_eval(DRO_13)


def testAngularDiameter():
    print(" ")
    print("ANGULAR DIAMETER TESTS")
    passes = 0
    angle = 2
    A = 500000
    [time,ang_diam_frac] = DRO_11.angular_diameter(angle,A,'N')

    # cross-check specific values outputted by function with values caluclated using Desoms
    if ang_diam_frac[1] > 0.454 and ang_diam_frac[1] < 0.457 and ang_diam_frac[100] > 0.493 and ang_diam_frac[100] < 0.496:
        print("DRO 11 Test Passed") 
        passes = passes + 1
    else:
        print("DRO 11 TEST FAILED")

    angle = 6
    [time,ang_diam_frac] = DRO_13.angular_diameter(angle,A,'N')
    if ang_diam_frac[1] == 1:
        print("Full view test passed") # check to see if angular diameter function works when entire view is captured
        passes = passes + 1
    else:
        print("FULL VIEW TEST FAILED")

    # conduct additional test for DRO 13
    angle = 1
    [time,ang_diam_frac] = DRO_13.angular_diameter(angle,A,'N')
    if ang_diam_frac[50] > 0.283 and ang_diam_frac[50] < 0.285:
        print("DRO 13 test passed")
        passes = passes + 1
    else:
        print("DRO_13 TEST FAILED")

    if passes == 3: # passes should equal total number of tests
        print("ALL ANGULAR DIAMETER TESTS PASSED")


    return

testAngularDiameter()