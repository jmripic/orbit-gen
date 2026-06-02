# The function in this file tests the orbit_volume method included in the Starlift_metrics class.
# It accounts for each kind of scenario that can be encountered when performing volume calculations
# in this context. The results of the function are checked against known values obtained by evaluating
# the volume properties of different shapes in SolidWorks
# All tests pass as of 5/17/24

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

# load pickle file
DRO_11 = Starlift_metrics.load_pickle_file("orbitFiles/DRO_11.241_days.p")
DRO_11 = Starlift_metrics.orbit_eval(DRO_11)

def testVolume():
    # tests that the volumes outputted from orbit_volume are correct 
    # based on SolidWorks cross-references for possible scenarios
    # scenarios are:
    # collides
    # back altitude
    # front altitude
    # full view

    passes = 0
    angle = 2
    A = 500000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A,'N')
    if V_percent_viewed[1] > 0.066 and V_percent_viewed[1] < 0.0665:
        print("Collides scenario passed")
        passes = passes + 1
    else:
        print("Collides scenario failed")

    # back altitude scenario
    angle = 3.8
    A = 500000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A,'N')
    if V_percent_viewed[1] > 0.485 and V_percent_viewed[1] < 0.486:
        print("Back altitude scenario passed")
        passes = passes + 1
    else:
        print("Back altitude scenario failed")

    # front altitude scenario
    angle = 42
    A = 20000000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A,'N')
    if V_percent_viewed[1] > 0.974 and V_percent_viewed[1] < 0.976:
        print("Front altitude scenario passed")
        passes = passes + 1
    else:
        print("Front altitude scenario failed")

    # full view scenario
    angle = 6
    A = 500000
    [time,V_percent_viewed,r_mag_vec] = DRO_11.orbit_volume(angle,A,'N')
    if V_percent_viewed[1] > 0.721 and V_percent_viewed[1] < 0.723:
        print("Full view scenario passed")
        passes = passes + 1
    else:
        print("Full view scenario failed")

    if passes == 4:
        print("ALL VOLUME TEST SCENARIOS PASSED")

testVolume()

    
            
