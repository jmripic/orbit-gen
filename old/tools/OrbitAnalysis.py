import sys, os, pathlib
sys.path.insert(1, 'C:/Users/jackc/Desktop/SP2024/Starlift/starlift')

from Solution import Solution
import scipy.io


## Execute ##

path_str = "orbitFiles/L1_S_10.003_days.p"
# path_str = "orbitFiles/L1_S_8.3257_days.mat"

DRO = Solution(path_str)
# DRO.nondimensionalize()
DRO.plot_orbit()



