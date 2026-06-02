import numpy as np
import sys
import os
from pathlib import Path
from glob import glob
from astropy.time import Time
import astropy.units as u
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.insert(1, 'tools')
import unitConversion
import frameConversion
import orbitEOMProp
import plot_tools
import extractTools
import spiceypy as spice
import multiShooting as ms
import singleShooting as ss

plt.rcParams.update({'font.size': 22})
spice.furnsh("fullForce.txt")

showPlots = True
# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])

radiiSun = spice.bodvrd( 'Sun', 'RADII', 3 )[1][0]
radiiEarth = spice.bodvrd( 'Earth', 'RADII', 3 )[1][0]
radiiMoon = spice.bodvrd( 'Moon', 'RADII', 3 )[1][0]
radii = np.array([radiiMoon, radiiEarth, radiiSun])

tNum = 61119    # vernal equinox 2026, March 20
#tNum = 61212    # summer solstice 2026, June 21
#tNum = 61306    # autumnal equinox 2026, Sept 23
#tNum = 61395    # winter solstice 2026, Dec 21

t_start = Time(tNum, format='mjd', scale='utc')        # vernal equinox 2026, March 20

mu_star = gmMoon/(gmEarth + gmMoon)
#mu_star = 1.215059E-2

et_start = spice.str2et(t_start.iso)
rvMoon = spice.spkezr('Moon', et_start, 'J2000', 'None', 'Earth')[0]
Tp_m = spice.oscltx(rvMoon, et_start, gmEarth)[-1]
omega_m = 2*np.pi/Tp_m

# Initial condition in canonical units in rotating frame R [pos, vel]
data = np.load('L2_SouthernN.npz')
#data = np.load('/Users/gracegenszler/Documents/Research/starlift/orbits/DRO.npz')
#posvelt = data['ICs']
#states = posvelt[:,0:6]
#periods = posvelt[:,6]
#breakpoint()
states = data['states']
periods = data['periods']
mu_star = data['mu_star']
m1 = (1 - mu_star)
m2 = mu_star

Tp0 = unitConversion.convertTime_to_dim(periods).value

testInds = np.array([32, 33])
#butterFlyInds = np.array([5, 20, 43, 59, 81, 98, 120, 138, 177])           # Done
#droInds = np.array([58, 74, 89, 97, 105, 120, 136, 149, 151])              # Done
#l2NorthernInds = np.array([0, 20, 41, 61, 81, 102, 122, 143, 163, 178])    # Done
#l2SouthernInds = np.array([5, 23, 41, 64, 85, 102, 123, 143, 163, 183])    # Done
#l1SouthernIndsP = np.array([3, 8, 12, 14])                                 # Done
#l1SouthernIndsN = np.array([27, 10, 7, 3, 0])                              # Done
#l1NorthernIndsN = np.array([31, 27, 16, 14, 4, 0])                         # Done
#l1NorthernIndsP = np.array([0, 2, 7, 10, 13])                              # Done

#Tp_target = np.linspace(Tp0[0],Tp0[-1], 10)
Tp_target = Tp0[testInds]

offset = 0
orbFactor = 1

for kk in np.arange(0 ,len(Tp_target)):
    t_start = Time(tNum-offset, format='mjd', scale='utc')
    Tp_diff = np.abs(Tp0 - Tp_target[kk])
    Tp_ind = np.argwhere(min(Tp_diff) == Tp_diff)[0,0]
    Tp_ind = Tp_ind + 0

#    state_kk = posvelt[Tp_ind,:]
    state_kk = states[Tp_ind]
    period_kk = periods[Tp_ind]
#    timeDays = unitConversion.convertTime_to_dim(state_kk[6]).value
    timeDays = unitConversion.convertTime_to_dim(period_kk).value
    totTime = timeDays*np.ceil((45)/timeDays)
#    totTime = timeDays*5
#    totTime = 36*3 + unitConversion.convertTime_to_dim(period_kk).value
#    totTime = 365*100 + unitConversion.convertTime_to_dim(period_kk).value
    mainDir = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/naturalOrbit/' + str(timeDays) + '_days/'

    giveUp = False
    attempt100Ctr = 0
    while not giveUp:
        orbitDir = mainDir + 'startTime_' + str(t_start.value)
        
        crtbpFilePath = Path(orbitDir + '/CRTBPData.npz')
        crtbpTotFilePath = orbitDir + '/CRTBPDataTot.npz'
        if crtbpFilePath.is_file():
            crtbpData = np.load(crtbpFilePath, allow_pickle=True)
            crtbpTotData = np.load(crtbpTotFilePath, allow_pickle=True)
            
            posCRTBP_R_dim = crtbpData['RPs']
            posCRTBP_I_dim = crtbpData['IPs']
            velCRTBP_R_dim = crtbpData['RVs']
            timesCRTBP_mjd = crtbpData['Ts']
            
            posCRTBP_dimtot = crtbpTotData['IPs']
            velCRTBP_dimtot = crtbpTotData['IVs']
            timesCRTBP_dtot = crtbpTotData['Ts']*u.d
            orbs = crtbpTotData['Norbit']
            timesCRTBP_mjdtot = crtbpTotData['Tmjd']
        else:
#            freeVar0CRTBP_R = np.array([state_kk[0], state_kk[2], state_kk[4], state_kk[6]])
            freeVar0CRTBP_R = np.array([state_kk[0], state_kk[2], state_kk[4], period_kk])

            statesCRTBP_R, timesCRTBP_R = orbitEOMProp.statePropCRTBP_R(freeVar0CRTBP_R, mu_star)  # State is in the R frame
            posCRTBP_R = statesCRTBP_R[:, 0:3]
            velCRTBP_R = statesCRTBP_R[:, 3:6]

            # Convert from nondimensional units to dimensional
            posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_star, 0, 0])).to_value(u.km)
            velCRTBP_R_dim = unitConversion.convertVel_to_dim(velCRTBP_R).to_value(u.km/u.s)
            timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
            timesCRTBP_mjd = t_start + timesCRTBP_d
            etCRTBP_mjd = spice.str2et(timesCRTBP_mjd.iso)

            posCRTBP_I_dim = np.zeros((len(etCRTBP_mjd), 3))
            for ii in np.arange(len(etCRTBP_mjd)):
                Crv_R2I = spice.sxform('MCR','MCI',etCRTBP_mjd[ii])
                state_R = np.append(posCRTBP_R_dim[ii], velCRTBP_R_dim[ii])
                state_I = Crv_R2I@state_R
                posCRTBP_I_dim[ii,:] = state_I[0:3]
        
            orbs = int(np.round(totTime/timesCRTBP_d[-1].value))
            posTol = .01    # km
            velTol = 1E-6*orbs  # km/s
            
            timesCRTBP_dtot = timesCRTBP_d.copy()
            timesCRTBP_mjdtot = timesCRTBP_mjd.copy()
            posCRTBP_dimtot = posCRTBP_R_dim.copy()
            velCRTBP_dimtot = velCRTBP_R_dim.copy()
            for ii in np.arange(1,orbs):
                next_times = timesCRTBP_d[1:] + timesCRTBP_dtot[-1]
                timesCRTBP_dtot = np.append(timesCRTBP_dtot, next_times)
                
                next_mjd = timesCRTBP_d[1:] + timesCRTBP_mjdtot[-1]
                timesCRTBP_mjdtot = Time(np.append(timesCRTBP_mjdtot.value, next_mjd.value), format='mjd', scale='utc')

                posCRTBP_dimtot = np.vstack((posCRTBP_dimtot[:-1,:], posCRTBP_R_dim))
                velCRTBP_dimtot = np.vstack((velCRTBP_dimtot[:-1,:], velCRTBP_R_dim))
            if showPlots:
                ax2 = plt.figure().add_subplot(projection='3d')
                ax2.plot(posCRTBP_dimtot[:,0], posCRTBP_dimtot[:,1], posCRTBP_dimtot[:,2])
                ax2.set_title('Multiple Orbits Check')
                ax2.set_xlabel('X [km]', labelpad = 30)
                ax2.set_ylabel('Y [km]', labelpad = 30)
                ax2.set_zlabel('Z [km]', labelpad = 30)
                plt.show(block=False)
                plt.show()
                breakpoint()

            os.makedirs(orbitDir)
            np.savez(orbitDir+'/CRTBPData.npz', RPs = posCRTBP_R_dim, RVs = velCRTBP_R_dim, Ts = timesCRTBP_mjd, IPs = posCRTBP_I_dim, mu_star = mu_star)
            np.savez(orbitDir+'/CRTBPDataTot.npz', Ts = timesCRTBP_dtot, Tmjd = timesCRTBP_mjdtot, IPs = posCRTBP_dimtot, IVs = velCRTBP_dimtot, Norbit = orbs)

        patchRound = int(np.round((timesCRTBP_mjd[-1]-timesCRTBP_mjd[0]).value))
        
        posTol = 0.01    # km
        velTol = 0.0001*orbs  # km/s
        
        initialFFFilePath = Path(orbitDir + '/InitialFF.npz')
        apoluneDataFilePath = orbitDir + '/ApoluneData.npz'
        if os.path.exists(initialFFFilePath):
            initialFF = np.load(initialFFFilePath)
            apoluneData = np.load(apoluneDataFilePath)
            
            stateApo = initialFF['ICs']
            timeApo = initialFF['Ts']
#            orbs = initialFF['Norbit']
            N1 = initialFF['Npatch']
     
            rotatedStatesApo = apoluneData['RICs']
            inertialStatesApo = apoluneData['IICs']
            timesFullApo = apoluneData['Ts']
            updatedInitialStatesApoR = apoluneData['patchStatesR']
            updatedInitialStatesApo = apoluneData['patchStatesI']
            updatedInitialEpochsApo = apoluneData['patchTimes']
            
            posTol = 0.01    # km
            velTol = 0.0001*orbs  # km/s
        else:
            N1 = int(patchRound*orbs*orbFactor - 1)
            exitflag = 0
            patchCtr1 = 0
            while exitflag != 1:
                posvel, taus = ms.getPatches(N1, timesCRTBP_dtot, timesCRTBP_mjdtot, posCRTBP_dimtot, velCRTBP_dimtot)

                # R frame, relative to the moon (MCR frame)
                posCRTBPM = posvel[:, 0:3]

                if showPlots:
                    ax3 = plt.figure().add_subplot(projection='3d')
                    for ii in np.arange(N1):
                        ax3.scatter(posvel[ii,0], posvel[ii,1], posvel[ii,2], c='g', marker='o')
                    ax3.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r')
                    ax3.set_xlabel('X [km]', labelpad = 30)
                    ax3.set_ylabel('Y [km]', labelpad = 30)
                    ax3.set_zlabel('Z [km]', labelpad = 30)
                    ax3.set_title('Patch Points in Moon Centered Rotating Frame')
                    plt.show(block=False)
#                    plt.show()

                # Convert to MCI frame
                initialEphemerisEpoch = spice.str2et(t_start.iso)
                times_dim = (taus - t_start).to_value(u.s)
                initialEphemerisEpoches = initialEphemerisEpoch + times_dim
                Crv_R2I = spice.sxform('MCR','MCI',initialEphemerisEpoches)
                initialEphemerisMCI = np.zeros((len(times_dim), 6))
                for ii in np.arange(len(times_dim)):
                    initialEphemerisMCI[ii,:] = Crv_R2I[ii,:,:] @ posvel[ii]

                updatedInitialEpochs, updatedInitialStates, exitflag = ms.multipleShootingI(initialEphemerisEpoches, initialEphemerisMCI, posTol, velTol, GM)
                
                if exitflag != 1:
                    N1 = N1 + 1
                    patchCtr1 = patchCtr1 + 1
                if patchCtr1 > patchRound:
        #            flag100Years = False
                    break
        
            if exitflag == 1:
                # Plot in MCI and MCR
                if showPlots:
                    ax4 = plt.figure().add_subplot(projection='3d')
                    ax5 = plt.figure().add_subplot(projection='3d')
                    ax4.plot([], [], label="Ephemeris", c='b')
                    ax4.plot([], [], label="CRTBP", c='r-.')
                inertialStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                rotatedStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                updatedInitialStatesApoR = np.array([])
                timesFull = np.array([])
                for ii in np.arange(N1-1):
                    Ts = updatedInitialEpochs[ii:ii+2]
                    times, states = ms.statePropFFI(Ts, updatedInitialStates[ii,:], GM)
                        
                    inertialStates = np.vstack((inertialStates, states))
                    
                    Crv_I2R = spice.sxform('MCI','MCR',times)
                    rStates = np.zeros((len(times), 6))
                    for jj in np.arange(len(times)):
                        rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
                    
                    rotatedStates = np.vstack((rotatedStates, rStates))
                    timesFull = np.append(timesFull, times)
                    updatedInitialStatesApoR = np.append(updatedInitialStatesApoR, rStates[0,:])
                    
                    if showPlots:
                        ax4.plot(rStates[:, 0], rStates[:, 1], rStates[:, 2], 'b')
                        ax4.scatter(rStates[0,0], rStates[0,1], rStates[0,2], c='g', marker='o')
#                        ax4.scatter(rStates[-1,0], rStates[-1,1], rStates[-1,2], c='y', marker='*')
                        
                        ax5.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
                        ax5.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
                        ax5.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
                updatedInitialStatesApoR = np.append(updatedInitialStatesApoR, rStates[-1,:])
                
                stateApo = inertialStates[1,:].copy()
                timeApo = timesFull[0].copy()
                rotatedStatesApo = rotatedStates[1:,:].copy()
                inertialStatesApo = inertialStates[1:,:].copy()
                timesFullApo = timesFull.copy()
                updatedInitialStatesApo = updatedInitialStates.copy()
                updatedInitialEpochsApo = updatedInitialEpochs.copy()
                updatedInitialStatesApoR = np.reshape(updatedInitialStatesApoR, (N1,6))
            
            if patchCtr1 > patchRound:
                break
            else:
                np.savez(orbitDir+'/InitialFF.npz', ICs = stateApo, Ts = timeApo, Norbit = orbs, Npatch = N1)
                np.savez(orbitDir+'/ApoluneData.npz', RICs = rotatedStatesApo, IICs = inertialStatesApo, Ts = timesFullApo, patchStatesI = updatedInitialStatesApo, patchTimes = updatedInitialEpochsApo, patchStatesR = updatedInitialStatesApoR)
            
        print('Switching to perilune-to-perilune and minimizing the number of patch points')
                
        periluneDataFilePath = Path(orbitDir + '/PeriluneData.npz')
        if os.path.exists(periluneDataFilePath):
            periluneData = np.load(periluneDataFilePath)
            
            rotatedStates = periluneData['RICs']
            inertialStates = periluneData['IICs']
            timesFull = periluneData['Ts']
            updatedInitialStatesPeriR = periluneData['patchStatesR']
            updatedInitialStates = periluneData['patchStatesI']
            updatedInitialEpochs = periluneData['patchTimes']
            NOrbPeri = periluneData['Norbit']
            orbs = NOrbPeri + 1
            N2 = periluneData['Npatch']
            
            posTol = 0.01    # km
            velTol = 0.0001*orbs  # km/s
            
            rmag = np.linalg.norm(rotatedStatesApo[1:,0:3], axis=1)
        else:
            apoluneData = np.load(apoluneDataFilePath)
            updatedInitialStates = apoluneData['patchStatesI']
            updatedInitialEpochs = apoluneData['patchTimes']
            rmag = np.linalg.norm(rotatedStatesApo[1:,0:3], axis=1)

            if showPlots:
                ax4.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.')
                ax4.set_xlabel('X [km]', labelpad = 30)
                ax4.set_ylabel('Y [km]', labelpad = 30)
                ax4.set_zlabel('Z [km]', labelpad = 30)
                ax4.set_title('Moon Centered Rotating Frame')
                plt.legend()
                plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

                ax5.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
                ax5.set_xlabel('X [km]')
                ax5.set_ylabel('Y [km]')
                ax5.set_zlabel('Z [km]')
                ax5.set_title('Moon Centered Inertial Frame')
                plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

#                fig, (ax6, ax7, ax8) = plt.subplots(3, 1)
#                ax6.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,0])
#                ax6.set_ylabel('x')
#                ax7.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,1])
#                ax7.set_ylabel('y')
#                ax8.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,2])
#                ax8.set_ylabel('z')

                plt.figure(9)
                plt.plot(np.arange(len(rotatedStatesApo[1:,2])), rmag)
                plt.show(block=False)

            # calculate all the perilunes
            rmags = np.linalg.norm(rotatedStatesApo[1:,0:3], axis=1)

            # find first perilune (new first patch point)
            oneOrb = (timesCRTBP_mjd[-1] - timesCRTBP_mjd[0]).to_value(u.s)
            difference_array1 = np.absolute((timesFullApo - timesFullApo[0]) - oneOrb)
            oneOrb_ind = difference_array1.argmin()
            rmag1 = np.linalg.norm(rotatedStatesApo[1:oneOrb_ind,0:3], axis=1)
            rmin1 = min(rmag1)
            min1_ind = np.argwhere(rmags == rmin1)[0][0]

            # find last perilune (new last patch point
            lastOrb = (orbs-1)*oneOrb
            difference_array2 = np.absolute((timesFullApo - timesFullApo[0]) - lastOrb)
            lastOrb_ind = difference_array2.argmin()
            rmagLast = np.linalg.norm(rotatedStatesApo[lastOrb_ind:,0:3], axis=1)
            rminLast = min(rmagLast)
            minLast_ind = np.argwhere(rmags == rminLast)[0][0]

            # redo the patches
            timesPeri_d = ((timesFullApo[min1_ind:minLast_ind] - timesFullApo[min1_ind])*u.s).to('d')
            t_startStr = spice.et2utc(timesFullApo[min1_ind], 'ISOC', 23, 24)
            t_startNew = Time(t_startStr, format='isot', scale='utc')
            timesPeri_mjd = Time(t_startNew.mjd + timesPeri_d.value, format='mjd', scale='utc')
            posPeri = inertialStatesApo[min1_ind:minLast_ind,0:3]
            velPeri = inertialStatesApo[min1_ind:minLast_ind,3:6]

            # decrease the number of patches until 2*(orbs - 1)
            velTol = 0.0001*(orbs - 1)  # km/s
            Nmax = int(np.round(patchRound*(orbs - 1)*orbFactor)) + 1
            Nmin = 2*(orbs - 1)
            Ns = np.append(np.arange(Nmax, Nmin, -(orbs-1)), Nmin)
            patchCtr = 0
            plusCtr = 0
            minPatch = False
            exitflag = 1
            minPatchN = np.array([len(updatedInitialEpochs)])
            while not minPatch:
                if exitflag == 1:
                    N2 = Ns[patchCtr]
                    plusCtr = 0
                else:
                    print('Solution not found. Increasing number of patch points by 1.')
                    plusCtr = plusCtr + 1
                    N2 = N2 + 1
                    print(plusCtr)
                    
                if plusCtr > patchRound:
                    breakpoint()
                    break
                posvel, taus = ms.getPatches(N2, timesPeri_d, timesPeri_mjd, posPeri, velPeri)
                initialEphemerisEpoch = spice.str2et(timesPeri_mjd[0].iso)
                times_dim = (taus - timesPeri_mjd[0]).to_value(u.s)
                initialEphemerisEpoches = initialEphemerisEpoch + times_dim

                updatedInitialEpochs, updatedInitialStates, exitflag = ms.multipleShootingI(initialEphemerisEpoches, posvel, posTol, velTol, GM)

                if exitflag == 1:
                    if np.any(Ns == N2):
                        patchCtr = patchCtr + 1
                        minPatchN = np.append(minPatchN, len(updatedInitialEpochs))
                        
                        if len(minPatchN) != len(np.unique(minPatchN)):
                            patchCtr = patchCtr - 1
                        
                    if N2 < Ns[-2] and N2 >= Ns[-1]:
                        minPatch = True
                    elif plusCtr <= (Ns[patchCtr-1] - Ns[patchCtr]) and plusCtr > 0:
                        minPatch = True
                        
                    # Plot in MCI and MCR
                    if showPlots:
                        ax10 = plt.figure().add_subplot(projection='3d')
                        ax11 = plt.figure().add_subplot(projection='3d')
                        ax10.plot([], [], label="Ephemeris", c='b')
                        ax10.plot([], [], label="CRTBP", c='r-.')
                    inertialStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                    rotatedStates = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                    timesFull = np.array([])
                    updatedInitialStatesPeriR = np.array([])
                    for ii in np.arange(N2-1):
                        Ts = updatedInitialEpochs[ii:ii+2]
                        times, states = ms.statePropFFI(Ts, updatedInitialStates[ii,:], GM)
                            
                        inertialStates = np.vstack((inertialStates, states))
                        
                        Crv_I2R = spice.sxform('MCI','MCR',times)
                        rStates = np.zeros((len(times), 6))
                        for jj in np.arange(len(times)):
                            rStates[jj,:] = Crv_I2R[jj,:,:]@states[jj,:]
                        
                        rotatedStates = np.vstack((rotatedStates, rStates))
                        timesFull = np.append(timesFull, times)
                        updatedInitialStatesPeriR = np.append(updatedInitialStatesPeriR, rStates[0,:])
                        
                        if showPlots:
                            ax10.plot(rStates[:, 0], rStates[:, 1], rStates[:, 2], 'b')
                            ax10.scatter(rStates[0,0], rStates[0,1], rStates[0,2], c='g', marker='o')
#                            ax10.scatter(rStates[-1,0], rStates[-1,1], rStates[-1,2], c='y', marker='*')
                            
                            ax11.plot(states[:, 0], states[:, 1], states[:, 2], 'b', label='Multi Segment')
                            ax11.scatter(states[0,0], states[0,1], states[0,2], c='g', marker='o')
                            ax11.scatter(states[-1,0], states[-1,1], states[-1,2], c='y', marker='*')
                    
                    diff = rotatedStates[-1] - rotatedStates[1]
                    rmag = np.linalg.norm(rotatedStates[1:,0:3], axis=1)
                    
                    if showPlots:
                        ax10.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.')
                        ax10.set_xlabel('X [km]', labelpad = 30)
                        ax10.set_ylabel('Y [km]', labelpad = 30)
                        ax10.set_zlabel('Z [km]', labelpad = 30)
                        ax10.set_title('Moon Centered Rotating Frame')
                        plt.figure(10)
                        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

                        ax11.plot(posCRTBP_I_dim[:,0], posCRTBP_I_dim[:,1], posCRTBP_I_dim[:,2], 'r-.', label='CRTBP')
                        ax11.set_xlabel('X [km]')
                        ax11.set_ylabel('Y [km]')
                        ax11.set_zlabel('Z [km]')
                        ax11.set_title('Moon Centered Inertial Frame')
                        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), borderaxespad=0)

#                        fig, (ax12, ax13, ax14) = plt.subplots(3, 1)
#                        ax12.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,0])
#                        ax12.set_ylabel('x')
#                        ax13.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,1])
#                        ax13.set_ylabel('y')
#                        ax14.plot(np.arange(len(rotatedStates[1:,2])),rotatedStates[1:,2])
#                        ax14.set_ylabel('z')

                        plt.figure(15)
                        plt.plot(np.arange(len(rotatedStates[1:,2])), rmag)
                        plt.show(block=False)
                    
                    # redo the patches
                    timesPeri_d = ((timesFull - timesFull[0])*u.s).to('d')
                    t_startStr = spice.et2utc(timesFull[0], 'ISOC', 23, 24)
                    t_startNew = Time(t_startStr, format='isot', scale='utc')
                    timesPeri_mjd = Time(t_startNew.mjd + timesPeri_d.value, format='mjd', scale='utc')
                    posPeri = inertialStates[1:,0:3]
                    velPeri = inertialStates[1:,3:6]
                    
            updatedInitialStatesPeriR = np.reshape(updatedInitialStatesPeriR, (N2-1,6))
            inertialStates = inertialStates[1:,:]
            rotatedStates = rotatedStates[1:,:]
            
            np.savez(orbitDir+'/PeriluneData.npz', RICs = rotatedStates, IICs = inertialStates, Ts = timesFull, patchStatesI = updatedInitialStates, patchTimes = updatedInitialEpochs, patchStatesR = updatedInitialStatesPeriR, Norbit = orbs-1, Npatch = N2)
            
        print('Minimum number of patch points for '+str(orbs-1)+' orbits: '+str(N2))
        plt.show()
        t100_initial = updatedInitialEpochs[0].copy()
        state100_initial = updatedInitialStates[0].copy()
        
        eventFilePath = orbitDir+'/attemptStartTime_'+str(t100_initial)+'*'
        
        oldFile = True      # ** Toggle this to False for GNSS
        advanceTime = (7*u.d).to_value(u.s)
        nFiles = len(os.listdir(orbitDir)) - 5
        fileCtr = 0
        while oldFile and (fileCtr <= nFiles):
            if glob(eventFilePath):
                t100_star = t100_initial + advanceTime
            
                t100_diff = np.abs(timesFull - t100_star)
                t100_min = min(t100_diff)
                t100_ind = np.argwhere(t100_diff == t100_min)[0,0]
                
                t100_initial = timesFull[t100_ind]
                eventFilePath = orbitDir+'/attemptStartTime_'+str(t100_initial)+'*'
                print('Attempt exists for ' + eventFilePath)
                fileCtr = fileCtr + 1
            else:
                oldFile = False
                print('Attempt does not exist')
        # ** Add oldFile = True for GNSS
        if not oldFile:
            solutionFound = False
            giveUp = False
            
            radii = np.append(radii, max(rmag))
            ms.hitMoon.terminal = True
            ms.hitEarth.terminal = True
            ms.hitSun.terminal = True
            ms.lostShape.terminal = True
            
            while not solutionFound:
                propTimes = np.array([])

                t100_final = t100_initial + (100.1*u.yr).to_value(u.s)
                
                sol_int = solve_ivp(ms.ffInertial, [t100_initial, t100_final], state100_initial, args=(GM,radii), events=[ms.hitMoon, ms.hitEarth, ms.hitSun, ms.lostShape], rtol=1E-12, atol=1E-12, method='LSODA')
                time100 = sol_int.t
                states100 = sol_int.y.T
                time100Events = sol_int.t_events
                states100Events = sol_int.y_events
                
                Crv_I2R = spice.sxform('MCI','MCR',time100)
                statesR100 = np.zeros((len(time100), 6))
                for ii in np.arange(len(time100)):
                    statesR100[ii,:] = Crv_I2R[ii,:,:]@states100[ii,:]
                
                if np.any(states100Events[0]):
                    print('Impacted Moon')
                    firstEventTime = time100Events[0][0]
                    firstEventState = states100Events[0][0]
                    firstEventInd = np.argwhere(time100 == firstEventTime)[0,0]
                    firstEventFlag = 0
                elif np.any(states100Events[1]):
                    print('Impacted Earth')
                    firstEventTime = time100Events[1][0]
                    firstEventState = states100Events[1][0]
                    firstEventInd = np.argwhere(time100 == firstEventTime)[0,0]
                    firstEventFlag = 1
                elif np.any(states100Events[2]):
                    print('Impacted Sun')
                    firstEventTime = time100Events[2][0]
                    firstEventState = states100Events[2][0]
                    firstEventInd = np.argwhere(time100 == firstEventTime)[0,0]
                    firstEventFlag = 2
                elif np.any(states100Events[3]):
                    print('Lost orbit shape')
                    firstEventTime = time100Events[3][0]
                    firstEventState = states100Events[3][0]
                    firstEventInd = np.argwhere(time100 == firstEventTime)[0,0]
                    firstEventFlag = 3
                else:
                    print('Solution Found')
                    firstEventTime = time100[-1]
                    firstEventState = states100[-1,:]
                    firstEventInd = len(time100)-1
                    firstEventFlag = -1
                
                propTime = ((firstEventTime-t100_initial)*u.s).to_value(u.d)
                print('Propagated for '+str(propTime)+' days')
                
                if showPlots:
                    ax16 = plt.figure().add_subplot(projection='3d')
                    ax16.plot(states100[:,0], states100[:,1], states100[:,2])
                    ax16.set_title(str((time100[-1]-time100[0])/60/60/24/365)+' year propagation')
                    ax16.set_xlabel('x [km]')
                    ax16.set_ylabel('y [km]')
                    ax16.set_zlabel('z [km]')
                    plt.show(block=False)
                
                np.savez(orbitDir+'/attemptStartTime_'+str(t100_initial)+'_propTime_'+str(propTime)+'.npz', states = states100, statesR = statesR100, Ts = time100, eventFlag = firstEventFlag)
                                
                if propTime > 365*5:
                    solutionFound = True
                    giveUp = True
                else:
                    t100_star = t100_initial + advanceTime
                    
                    t100_diff = np.abs(timesFull - t100_star)
                    t100_min = min(t100_diff)
                    t100_ind = np.argwhere(t100_diff == t100_min)[0,0]
                    
                    t100_initial = timesFull[t100_ind]
                    state100_initial = inertialStates[t100_ind,:]
                    
                    if t100_ind == (len(timesFull) - 1):
                        attempt100Ctr = attempt100Ctr + 1
                        t_start = Time(t_start.value + (updatedInitialEpochs[-1] - updatedInitialEpochs[0])/60/60/24, format='mjd', scale='utc')
                        # break out of the solutionFound while loop
                        break
        else:
            attempt100Ctr = attempt100Ctr + 1
            t_start = Time(t_start.value + (updatedInitialEpochs[-1] - updatedInitialEpochs[0])/60/60/24, format='mjd', scale='utc')
        
        if showPlots:
            plt.close('all')
        
        if attempt100Ctr >= 1:      # ** CHANGE BACK TO 1 for GNSS
            giveUp = True
            
#        breakpoint()

breakpoint()
