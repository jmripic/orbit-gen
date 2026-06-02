import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, 'tools')
import orbitEOMProp
import pdb

guess = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0,  1.3632096570/2]
mu_star = 1.215059*10**(-2)
N = 10
ICs = orbitEOMProp.generateFamily_CRTBP(guess, mu_star, N)

ax = plt.figure().add_subplot(projection='3d')
for ii in np.arange(N):
    ax.plot(ICs[ii, 0], ICs[ii, 1], ICs[ii, 2])

plt.show()
