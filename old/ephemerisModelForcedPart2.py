import numpy as np
import sys
import os
import astropy.coordinates as coord
from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from scipy.io import loadmat
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
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
import csv
plt.rcParams.update({'font.size': 22})
spice.furnsh("fullForce.txt")

# Parameters
gmSun = spice.bodvrd( 'Sun', 'GM', 1 )[1][0]
gmEarth = spice.bodvrd( 'Earth', 'GM', 1 )[1][0]
gmMoon = spice.bodvrd( 'Moon', 'GM', 1 )[1][0]
GM = np.array([gmMoon, gmEarth, gmSun])

#fileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/naturalOrbit/9.081994780576505_days/startTime_61119.0/'
#fileStr = 'naturalOrbit'
fileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/'
#fileStr = 'L1_Halo'                     # L1 Halo
#fileStr = 'L2_NRHO'                     # L2 NRHO
fileStr = 'TrajI_1265_MassOptimal'      # L2 Halo
#fileStr = 'TrajI_1265_EnergyOptimal'    # L2 Halo
#fileStr = 'L2_Butterfly'                # L2 Butterfly
#fileStr = 'TrajExample'                 # pole sitter
folders = [fileStr+'/']

for jj in np.arange(len(folders)):
    data = np.load(fileDirectory+folders[jj]+'InitialFF.npz', allow_pickle=True)

    t_start = data['startTime']+0
    patchStates = data['ICs']
    patchStatesFinal = data['FCs']
    patchTimes = data['Ts']
    statesR = data['statesR'][1:,:]
    statesI = data['statesI'][1:,:]
    timesFF = data['times']
    N = data['Npatch']
    dVtot = data['dVpatches']
    mu_cstar = 0.01215059
    
#    data = np.load(fileDirectory+'ApoluneData.npz', allow_pickle=True)
#    
#    patchStates = data['patchStatesR']
#    patchStatesFinal = patchStates[1:,:]
#    patchTimes = data['patchTimes']
#    statesR = data['RICs'][1:,:]
#    statesI = data['IICs'][1:,:]
#    timesFF = data['Ts']
#    N = np.shape(patchStates)[0]
#    dVtot = sum(np.linalg.norm(patchStates[:,3:6],axis=1))
#    t_start = timesFF[0]
#    mu_cstar = gmMoon/(gmMoon+gmEarth)

#    et_start = t_start
    et_start = spice.str2et(t_start.iso)
    rvMoon = spice.spkezr('Moon', et_start, 'J2000', 'None', 'Earth')[0]
    Tp_m = spice.oscltx(rvMoon, et_start, gmEarth)[-1]
    omega_m = 2*np.pi/Tp_m

    mat_data = loadmat(fileStr+'.mat')['TrajI']
    posCRTBP_R = mat_data[:,0:3]
    velCRTBP_R = mat_data[:,3:6]
    timesCRTBP_R = mat_data[:,6]
    uT = mat_data[:,7:]

#    posCRTBP_R = statesR[:,0:3]
#    velCRTBP_R = statesR[:,3:6]
#    timesCRTBP_R = timesFF
#    uT = np.zeros((len(timesFF), 3))

    posCRTBP_R_dim = unitConversion.convertPos_to_dim(posCRTBP_R - np.array([1-mu_cstar, 0, 0])).to_value(u.km)
    velCRTBP_R_dim = unitConversion.convertVel_to_dim(velCRTBP_R).to_value(u.km/u.s)
    timesCRTBP_d = unitConversion.convertTime_to_dim(timesCRTBP_R).to('d')
    timesCRTBP_mjd = t_start + timesCRTBP_d
    etCRTBP_mjd = spice.str2et(timesCRTBP_mjd.iso)
#    etCRTBP_mjd = timesCRTBP_mjd
    uT_dim = unitConversion.convertAcc_to_dim(uT).to('km/s**2')

    # Calculate delta-v
    uT_mag = np.linalg.norm(uT_dim, axis=1)
    dVCRTBPtot = cumulative_trapezoid(uT_mag, x=etCRTBP_mjd, axis=0)*u.km/u.s
    dVCRTBP = np.append(0*u.km/u.s,dVCRTBPtot)
    dVCRTBP = np.diff(dVCRTBP)
    
    dStates = patchStatesFinal - patchStates[1:,:]
    Crv_I2R = spice.sxform('MCI','MCR',patchTimes)
    dStatesR = np.zeros((len(patchTimes)-2, 6))
    for kk in np.arange(1,len(patchTimes)-1):
        dStatesR[kk-1,:] = Crv_I2R[kk,:,:]@dStates[kk-1,:]
#    # Default
    mi = 1000*u.kg                          # kg
    g0 = const.g0.value*const.g0.unit       # m/s^2
    Isp = 1500*u.s
    Ftmax0 = (max(uT_mag)*mi).to_value(u.mN)
#    Ftmax0 = 50
    
#    thrusterName = "baseline"
#    Isps = np.array([Isp.value])*u.s
#    Ftmaxs = np.array([Ftmax0])*u.mN
    
#    thrusterName = "ST-100 Hall Thruster"      # https://sets.space/wp-content/themes/sets-space/images/product-sheet/2023/ST-100.pdf
#    Ftmax = Ftmax0*u.mN
#    Ftmax0 = (max(uT_mag)*mi).to_value(u.mN)
##    Ftmaxs = np.array([Ftmax0, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 107])*u.mN
#    Isps = np.array([1500, 1550, 1600, 1650, 1700])*u.s

    thrusterName = "BHT-1500 Hall Thruster"    # https://www.busek.com/bht1500
#    Ftmaxs = np.array([68, 101, 120, 134, 158, 179])*u.mN
#    Isps = np.array([1615, 1710, 1740, 1700, 1735, 1865])*u.s
    Ftmaxs = np.array([58, 87, 103, 118, 143, 154])*u.mN
    Isps = np.array([1860, 1895, 1940, 1915, 2045, 2035])*u.s

#    thrusterName = 'varyMass'
#   thrusterName = "ST-25 Hall Thruster"        # https://sets.space/wp-content/themes/sets-space/images/product-sheet/2023/ST-25.pdf   0, 3
#   thrusterName = "BHT-100 Hall Thruster"      # https://www.busek.com/bht100-hall-thruster 1
#   thrusterName = "BHT-200 Hall Thruster"      # https://www.busek.com/bht200 2
#   thrusterName = "ST-40 Hall Thruster"        # https://sets.space/wp-content/themes/sets-space/images/product-sheet/2023/ST-40.pdf   4, 6
#   thrusterName = "BHT-350 Hall Thruster"      # https://www.busek.com/bht350 5
#   thrusterName = "BHT-600 Hall Thruster"      # https://www.busek.com/bht600 7
#   thrusterName = "BHT-6000 Hall Thruster"     # https://www.busek.com/bht6000 9, 10
#   thrusterName = "BHT-20k Hall Thruster"      # https://www.busek.com/bht20k-hall-thruster 11
#    Ftmaxs = np.array([5, 7, 13, 14, 15, 17, 34, 39, Ftmax0, 298, 325, 1006])*u.mN
#    Isps = np.array([1000, 1000, 1390, 1500, 1300, 1244, 1750, 1500, 1500, 2708, 2029, 2515])*u.s

    ff = open(fileDirectory+folders[jj]+thrusterName+'varyIspTMP2.txt', 'w')
#    ff = open(fileDirectory+thrusterName+'naturalOrb.txt', 'w')
    ff.write(thrusterName+"\n")
#    for kk in np.arange(len(Ftmaxs)):
    for kk in np.arange(len(Isps)):
        Ftmax = Ftmaxs[kk]
        Isp = Isps[kk]
#        mi = mis[kk]
        
        mf = mi*np.exp(-dVCRTBPtot/(Isp*g0))    # kg
        m_dim = np.append(mi, mf)               # mass history

        # Convert impulsive burns into continuous thrust
        sigma = .9999
        sigma2 = 1.000
        inds = np.array([])
        dts = np.array([])
        Ups = np.array([])
        # review this to figure out correct value to save to Ups, currently adding existing control twice in the end
        for ii in np.arange(1,len(patchTimes)-1):
            uT_flag = False
            timeDiff1 = np.abs(etCRTBP_mjd - patchTimes[ii])
            
            ind_ii = np.argwhere(timeDiff1 == min(timeDiff1))[0,0]
            uT_ii = uT_mag[ind_ii]
            m_ii = m_dim[ind_ii]
    #        m_ii = mi
            
            while not uT_flag:
                F_ii = uT_ii*m_ii
#                tmpFt = F_ii.to_value(u.mN)*1000
                tmpFt = Ftmax0
                if tmpFt >= Ftmax0:
                    Ftmax = Ftmax0*u.mN
                else:
                    Ftmax = tmpFt*u.mN
                    
                F_burn = Ftmax*sigma - F_ii
                uT_burn = F_burn/m_ii
                
                dV_burn = (np.linalg.norm(dStatesR[ii-1,3:6])*(u.km/u.s))
                dt_burn = dV_burn/uT_burn

                if dt_burn < 0:
                    breakpoint()
                # check the uT in the dt range, centered at patch point
                t_initial = patchTimes[ii] - dt_burn.value/2
                t_final = patchTimes[ii] + dt_burn.value/2
                
                timeDiff2 = np.abs(etCRTBP_mjd - t_initial)
                timeDiff3 = np.abs(etCRTBP_mjd - t_final)
                ind_i = np.argwhere(timeDiff2 == min(timeDiff2))[0,0]-1
                ind_f = np.argwhere(timeDiff3 == min(timeDiff3))[0,0]+1
                
                uTcheck = np.argwhere(F_burn + uT_mag[ind_i:ind_f]*m_dim[ind_i:ind_f] > Ftmax)

                if len(uTcheck) > 0:
                    # rescale time based off of max thrust force in current time interval
                    uT_ii = max(uT_mag[ind_i:ind_f])*sigma2
        
                    m_ii = mi*np.exp(-(dV_burn + dVCRTBPtot[ind_i])/(Isp*g0))    # kg
                else:
                    # save info to create continuous burn
                    inds = np.append(inds, np.array([ind_i, ind_f]))
                    dts = np.append(dts, (dt_burn.to('s')).value)
                    Ups = np.append(Ups, (uT_burn.to('km/s**2')).value)
                    uT_flag = True

            dt_print_s = dt_burn.to('s')
            if dt_print_s.value < 60:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_print_s))
            elif dt_print_s.value < 60*60:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_burn.to('min')))
            elif dt_print_s.value < 60*60*24:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_burn.to('hr')))
            else:
                print('Burn duration for patch '+str(ii)+' is: '+str(dt_burn.to('d')))
        
        # create a new uT array
        t_min = patchTimes[1] - dts[0]/2
        t_max = patchTimes[1] + dts[0]/2
        
        ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
        ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
        uT_burn = Ups[0]*dStatesR[0,3:6]/np.linalg.norm(dStatesR[0,3:6])

        etCRTBP_mjd_new = etCRTBP_mjd[:ind_min].copy()
        uT_new = uT_dim[:ind_min,:].copy().value

        points_uT = (etCRTBP_mjd, np.array([0, 1, 2]))
        interp_fc_uT = RegularGridInterpolator(points_uT, uT_dim, method='linear')

        if ind_old - ind_min >= 1:
            uT_patch = uT_burn + uT_dim[ind_min+1:ind_old,:].value
            uT_patch_i = uT_burn + interp_fc_uT((t_min, np.array([0, 1, 2])))
            uT_patch_f = uT_burn + interp_fc_uT((t_max, np.array([0, 1, 2])))
            uT_patch = np.vstack((np.vstack((uT_patch_i, uT_patch)), uT_patch_f))

            et_patch = np.append(np.append(t_min, etCRTBP_mjd[ind_min+1:ind_old]), t_max)
            etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, et_patch)
        else:
            uT_patch = uT_burn + uT_dim[ind_min,:].value
            uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
            etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, patchTimes[1], t_max]))

        uT_new = np.vstack((uT_new,uT_patch))
        for ii in np.arange(1,len(dts)):
            t_min = patchTimes[ii+1] - dts[ii]/2
            t_max = patchTimes[ii+1] + dts[ii]/2
            
            ind_min = np.argwhere(etCRTBP_mjd < t_min)[-1,0]
            ind_max = np.argwhere(etCRTBP_mjd > t_max)[0,0]
            uT_burn = Ups[ii]*dStatesR[ii,3:6]/np.linalg.norm(dStatesR[ii,3:6])

            etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:ind_min])
            uT_new = np.vstack((uT_new, uT_dim[ind_old:ind_min,:].value))
            if ind_max - ind_min > 1:
                uT_patch = uT_burn + uT_dim[ind_min:ind_max,:].value
                uT_patch_i = uT_burn + interp_fc_uT((t_min, np.array([0, 1, 2])))
                uT_patch_f = uT_burn + interp_fc_uT((t_max, np.array([0, 1, 2])))
                uT_patch = np.vstack((np.vstack((uT_patch_i, uT_patch)), uT_patch_f))

                et_patch = np.append(np.append(t_min, etCRTBP_mjd[ind_min:ind_max]), t_max)
                etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, et_patch)
            else:
                uT_patch = uT_burn + uT_dim[ind_min,:].value
                uT_patch = np.reshape(np.repeat(uT_patch, repeats=3,axis=0),(3,3)).T
                
                etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, np.array([t_min, patchTimes[ii+1], t_max]))
            uT_new = np.vstack((uT_new,uT_patch))

            ind_old = np.argwhere(etCRTBP_mjd > t_max)[0,0]
        uT_new = np.vstack((uT_new, uT_dim[ind_old:,:].value))
        etCRTBP_mjd_new = np.append(etCRTBP_mjd_new, etCRTBP_mjd[ind_old:])

        Ts = np.array([patchTimes[0], patchTimes[-1]])

        times_final, states_final = ms.statePropFFIForced(Ts, patchStates[0,:], GM, uT_new, etCRTBP_mjd_new, omega_m)

        states_final_R = np.zeros((len(times_final), 6))
        for ii in np.arange(len(times_final)):
            Crv_I2R = spice.sxform('MCI','MCR',times_final[ii])
            states_final_R[ii,:] = Crv_I2R@states_final[ii,:]
            
        ax1 = plt.figure(figsize=(16, 12)).add_subplot(projection='3d')
        ax1.plot(statesR[:, 0], statesR[:, 1], statesR[:, 2], 'b', label='Multi Segment - Ephemeris Model')
        ax1.plot(posCRTBP_R_dim[:,0], posCRTBP_R_dim[:,1], posCRTBP_R_dim[:,2], 'r-.', label='Single Segment - CRTBP')
        ax1.plot(states_final_R[:,0], states_final_R[:,1], states_final_R[:,2], 'y-.', label='Final Trajectory')
        ax1.set_xlabel('X [km]', labelpad = 30)
        ax1.set_ylabel('Y [km]', labelpad = 30)
        ax1.set_zlabel('Z [km]', labelpad = 30)
        ax1.set_title('Moon Centered Rotating Frame')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')

        points_R = (times_final, np.array([0, 1, 2]))
        interp_fc_R = RegularGridInterpolator(points_R, states_final_R[:,0:3], method='linear')

        statesR_interp = np.zeros((len(timesFF), 3))
        for ii in np.arange(len(timesFF)):
            statesR_interp[ii,:] = interp_fc_R((timesFF[ii], np.array([0, 1, 2])))

        finalR_mag = np.linalg.norm(statesR[:,0:3], axis = 1)
        interpR_mag = np.linalg.norm(statesR_interp[:,0:3], axis = 1)

        statesR_diff = statesR_interp[:,0:3] - statesR[:,0:3]
        diffR_mag = interpR_mag - finalR_mag
        print('Max deviation is '+str(max(diffR_mag))+' km')

        burnTimeTot = sum(dts)
        if burnTimeTot < 60:
            print('Total burn time is: '+str(burnTimeTot)+' s')
        elif burnTimeTot < 60*60:
            print('Total burn time is: '+str(burnTimeTot/60)+' min')
        elif burnTimeTot < 60*60*24:
            print('Total burn time is: '+str(burnTimeTot/60/60)+' hr')
        else:
            print('Total burn time is: '+str(burnTimeTot/60/60/24)+' d')
        
        statesR_diff = statesR_diff
        plot_time = (timesFF - timesFF[0])/60/60/24
        fig, (ax2, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(10, 8))
#        ax2.plot(plot_time, abs(statesR_diff[:,0]))
#        ax2.set_ylabel('x [km]')
#        ax2.set_xlim(0, plot_time[-1])
#        ax2.get_xaxis().set_visible(False)
#        ax2.set_title('Absolute Value Differences')
#        ax3.plot(plot_time, abs(statesR_diff[:,1]))
#        ax3.set_ylabel('y [km]')
#        ax3.set_xlim(0, plot_time[-1])
#        ax3.get_xaxis().set_visible(False)
#        ax4.plot(plot_time, abs(statesR_diff[:,2]))
#        ax4.set_ylabel('z [km]')
#        ax4.set_xlim(0, plot_time[-1])
#        ax4.get_xaxis().set_visible(False)
#        ax5.plot(plot_time, abs(diffR_mag))
#        ax5.set_ylabel('Position Magnitude [km]')
##        ax5.set_ylabel('Position Magnitude [km]', labelpad = 10)
#        ax5.set_xlabel('Time [days]')
#        ax5.set_xlim(0, plot_time[-1])
        ax2.plot(plot_time, abs(statesR_diff[:,0])*100)
        ax2.set_ylabel('x [m]')
        ax2.set_xlim(0, plot_time[-1])
        ax2.get_xaxis().set_visible(False)
        ax2.set_title('Absolute Value Differences')
        ax3.plot(plot_time, abs(statesR_diff[:,1])*100)
        ax3.set_ylabel('y [m]')
        ax3.set_xlim(0, plot_time[-1])
        ax3.get_xaxis().set_visible(False)
        ax4.plot(plot_time, abs(statesR_diff[:,2])*100)
        ax4.set_ylabel('z [m]')
        ax4.set_xlim(0, plot_time[-1])
        ax4.get_xaxis().set_visible(False)
        ax5.plot(plot_time, abs(diffR_mag)*100)
        ax5.set_ylabel('Position Magnitude [m]')
#        ax5.set_ylabel('Position Magnitude [km]', labelpad = 10)
        ax5.set_xlabel('Time [d]')
        ax5.set_xlim(0, plot_time[-1])
        
        uTNew_mag = np.linalg.norm(uT_new, axis=1)*u.km/u.s**2
        dVCRTBPtotNew = cumulative_trapezoid(uTNew_mag, x=etCRTBP_mjd_new, axis=0)*u.km/u.s
        dVCRTBPNew = np.append(0*u.km/u.s,dVCRTBPtotNew)
        dVCRTBPNew = np.diff(dVCRTBPNew)
        mfNew = mi*np.exp(-dVCRTBPtotNew/(Isp*g0))    # kg
        mNew_dim = np.append(mi, mfNew)               # mass history
        
        print('Final mass is : '+str(mNew_dim[-1]))
            
        ff.write(str(Ftmax.to_value(u.mN))+", "+str(Isp.value)+", "+str(burnTimeTot)+", "+str(max(diffR_mag))+", "+str(mi.value)+", "+str(mNew_dim[-1].value)+"\n")
        
        uTNew_time = ((etCRTBP_mjd_new-etCRTBP_mjd_new[0])*u.s).to_value(u.d)
        FtMaxPlt = (Ftmax).to_value(u.mN)*np.array([1, 1])

#        plt.figure(6)
#        plt.plot(timesCRTBP_d.value, (uT_mag*m_dim).to_value(u.mN), 'b', label='Original Thrust Profile')
#        plt.plot(uTNew_time, (uTNew_mag*mNew_dim).to_value(u.mN), 'r-.', label='Recreated Thrust Profile')
#        plt.plot(np.array([timesCRTBP_d[0].value, timesCRTBP_d[-1].value]), FtMaxPlt, 'k', label='Max Thrust')
#        plt.xlabel('Time [d]')
#        plt.ylabel('Thrust Force [mN]')
#        plt.yscale('log')
#        plt.xlim(0, uTNew_time[-1])
#        plt.legend(bbox_to_anchor=(1.28, .84),loc='lower right')

        fig, (ax9, ax6) = plt.subplots(2, 1)
        ax9.plot(timesCRTBP_d.value, (uT_mag*m_dim).to_value(u.mN), 'b', label='Original Thrust Profile')
        ax9.plot(uTNew_time, (uTNew_mag*mNew_dim).to_value(u.mN), 'r-.', label='Recreated Thrust Profile')
        ax9.plot(np.array([timesCRTBP_d[0].value, timesCRTBP_d[-1].value]), FtMaxPlt, 'k', label='Max Thrust')
        ax9.set_ylabel('Thrust Force [mN]')
        ax9.set_xlim(0, uTNew_time[-1])
        ax9.set_ylim(49.8, 50.2)
        ax9.get_xaxis().set_visible(False)
        ax6.plot(timesCRTBP_d.value, (uT_mag*m_dim).to_value(u.mN), 'b', label='Original Thrust Profile')
        ax6.plot(uTNew_time, (uTNew_mag*mNew_dim).to_value(u.mN), 'r-.', label='Recreated Thrust Profile')
        ax6.plot(np.array([timesCRTBP_d[0].value, timesCRTBP_d[-1].value]), FtMaxPlt, 'k', label='Max Thrust')
        ax6.set_xlabel('Time [d]')
        ax6.set_ylabel('Thrust Force [mN]')
        ax6.set_xlim(0, uTNew_time[-1])
        plt.legend()
#        plt.show()
        
        plt.figure(7)
        plt.plot(uTNew_time, mNew_dim, label='Recreated Mass Profile')
        plt.plot(timesCRTBP_d.value, m_dim, label='Original Mass Profile')
        plt.legend()
        plt.xlabel('Time [d]')
        plt.ylabel('Mass [kg]')
        
        plt.figure(8)
        plt.plot(uTNew_time[:-1], dVCRTBPNew.to_value(u.m/u.s), 'b', label='Recreated Delta-v Profile')
        plt.plot(timesCRTBP_d[:-1].value, (dVCRTBP).to_value(u.m/u.s), 'r-.', label='Original Delta-v Profile')
        plt.scatter(((patchTimes[1:-1]-patchTimes[0])*u.s).to_value(u.d), dVtot, c='g', marker='*', zorder=3, label='Patch Point Burns')
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('delta-v [m/s]')
        plt.yscale('log')
        breakpoint()
#        plt.show()
#        breakpoint()
        plt.close('all')
    ff.close()
