import numpy as np
import collections
import os
import h5py
import matplotlib.pyplot as plt
import mpl_style
import sys
import time
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from mymodule.stats import percentiles
from mymodule.main_branch_info import read_dhalo_data, choose_sim, read_dhalo_data_subv, read_shark_data, read_shark_sfh
from mymodule.plotting import plot_prop, plot_prop_frac, plot_hist, plot_frac
import functools
import common
import utilities_statistics as us

plt.style.use(mpl_style.style1)
#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(suppress=True)

# line plotting properties
linw1 = 3.5
linw2 = 2
labs = 45

rem_sat = True
only_centrals = 3
only_centrals_hmf = 0
all_branches = True
shark_model = 'lagos23_ign'
allsubv = True
include_indv = False
include_obs = True
    
# values of exponential profiles
alpha_up = 3
alpha_down = 0.1
offset_up = 1
offset_down = -0.5


# constants
GyrToYr = 1e9
obsdir = '/home/achandro/external_codes/shark/data'
observation = collections.namedtuple('observation', 'label x y yerrup yerrdn err_absolute')
# bins
mlow = 5
mupp = 15.6
dm = 0.2
mbins = np.arange(mlow,mupp,dm)
xmf = mbins + dm/2.0

def load_smf_observations(obsdir, h0):
    # Thorne et al. (2021)
    def add_thorne21_data(zappend, file_read='z0.4'):
        lm, pD, dn, du = np.loadtxt(obsdir+'/mf/SMF/Thorne21/SMFvals_'+file_read+'.csv', delimiter=',', skiprows=1, usecols = [0,1,2,3], unpack = True)
        hobd = 0.7
        pDlog = np.log10(pD[::3]) +  3.0 * np.log10(hobs/h0)
        dnlog = np.log10(pD[::3]) - np.log10(dn[::3])
        dulog = np.log10(du[::3]) - np.log10(pD[::3])
        lm = lm[::3] -  2.0 * np.log10(hobs/h0)
        zappend.append((observation("Thorne+2021", lm, pDlog, dnlog, dulog, err_absolute=False), 's'))

    # Weaver et al. (2022; COSMOS2020)
    def add_weaver22_data(zappend, file_read='0.2z0.5'):
        lm, pD, dn, du = np.loadtxt(obsdir+'/mf/SMF/COSMOS2020/SMF_Farmer_v2.1_' + file_read + '_total.txt', delimiter=' ', skiprows=0, usecols = [0,2,3,4], unpack = True)
        hobd = 0.7
        pDlog = np.log10(pD) +  3.0 * np.log10(hobs/h0)
        dnlog = np.log10(pD) - np.log10(dn)
        dulog = np.log10(du) - np.log10(pD)
        lm = lm -  2.0 * np.log10(hobs/h0)
        zappend.append((observation("Weaver+2023", lm, pDlog, dnlog, dulog, err_absolute=False), '*'))


    # Driver al. (2022, z=0). Chabrier IMF
    z0obs = []
    lm, p, dp = common.load_observation(obsdir, 'mf/SMF/GAMAIV_Driver22.dat', [0,1,2])
    hobs = 0.7
    xobs = lm + 2.0 * np.log10(hobs/h0)
    yobs = p - 3.0 * np.log10(hobs/h0)
    z0obs.append((observation("Driver+2022", xobs, yobs, dp, dp, err_absolute=False), 'o'))

    lm, p, dpdn, dpup = common.load_observation(obsdir, 'mf/SMF/SMF_Bernardi2013_SerExp.data', [0,1,2,3])
    xobs = lm + 2.0 * np.log10(hobs/h0)
    yobs = np.log10(p) - 3.0 * np.log10(hobs/h0)
    ydn = np.log10(p) - np.log10(p-dpdn)
    yup = np.log10(p+dpup) - np.log10(p) 
    z0obs.append((observation("Bernardi+2013", xobs, yobs, ydn, yup, err_absolute=False), 's'))


    lm, p, dpdn, dpup = common.load_observation(obsdir, 'mf/SMF/SMF_Li2009.dat', [0,1,2,3])
    xobs = lm - 2.0 * np.log10(hobs) + 2.0 * np.log10(hobs/h0)
    yobs = p + 3.0 * np.log10(hobs) - 3.0 * np.log10(hobs/h0)
    z0obs.append((observation("Li&White+2009", xobs, yobs, abs(dpdn), dpup, err_absolute=False), 'd'))


    # Moustakas (Chabrier IMF), ['Moustakas+2013, several redshifts']
    zdnM13, lmM13, pM13, dp_dn_M13, dp_up_M13 = common.load_observation(obsdir, 'mf/SMF/SMF_Moustakas2013.dat', [0,3,5,6,7])
    xobsM13 = lmM13 + 2.0 * np.log10(hobs/h0)

    yobsM13 = np.full(xobsM13.shape, -999.) - 3.0 * np.log10(hobs/h0)
    lerrM13 = np.full(xobsM13.shape, -999.)
    herrM13 = np.full(xobsM13.shape, 999.)
    indx = np.where( pM13 < 1)
    yobsM13[indx] = (pM13[indx])
    indx = np.where( dp_dn_M13 > 0)
    lerrM13[indx]  = dp_dn_M13[indx] 
    indx = np.where( dp_up_M13 > 0)
    herrM13[indx]  = dp_up_M13[indx]

    # Muzzin (Kroupa IMF), ['Moustakas+2013, several redshifts']
    zdnMu13,zupMu13,lmMu13,pMu13,dp_dn_Mu13,dp_up_Mu13 = common.load_observation(obsdir, 'mf/SMF/SMF_Muzzin2013.dat', [0,1,2,4,5,5])
    # -0.09 corresponds to the IMF correction
    xobsMu13 = lmMu13 - 0.09 + 2.0 * np.log10(hobs/h0) 
    yobsMu13 = np.full(xobsMu13.shape, -999.) - 3.0 * np.log10(hobs/h0)
    lerrMu13 = np.full(xobsMu13.shape, -999.)
    herrMu13 = np.full(xobsMu13.shape, 999.)
    indx = np.where( pMu13 < 1)
    yobsMu13[indx] = (pMu13[indx])
    indx = np.where( dp_dn_Mu13 > 0)
    lerrMu13[indx]  = dp_dn_Mu13[indx] 
    indx = np.where( dp_up_Mu13 > 0)
    herrMu13[indx]  = dp_up_Mu13[indx]

    # z0.5 obs
    z05obs = []
    #in_redshift = np.where(zdnM13 == 0.4)
    #z05obs.append((observation("Moustakas+2013", xobsM13[in_redshift], yobsM13[in_redshift], lerrM13[in_redshift], herrM13[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zdnMu13 == 0.5)
    z05obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), 'o'))
    add_thorne21_data(z05obs, file_read='z0.51')
    add_weaver22_data(z05obs, file_read='0.2z0.5')


    # z1 obs
    z1obs = []
    #in_redshift = np.where(zdnM13 == 0.8)
    #z1obs.append((observation("Moustakas+2013", xobsM13[in_redshift], yobsM13[in_redshift], lerrM13[in_redshift], herrM13[in_redshift], err_absolute=False), 'o'))
    in_redshift = np.where(zdnMu13 == 1)
    z1obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), 'o'))
    add_thorne21_data(z1obs, file_read='z1.1')
    add_weaver22_data(z1obs, file_read='0.8z1.1')

    #z2 obs
    z2obs = []
    in_redshift = np.where(zupMu13 == 2.5)
    z2obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), 'o'))
    #in_redshift = np.where(zdnS12 == 1.8)
    #z2obs.append((observation("Santini+2012", xobsS12[in_redshift], yobsS12[in_redshift], lerrS12[in_redshift], herrS12[in_redshift], err_absolute=False), 'o'))
    add_thorne21_data(z2obs, file_read='z2')
    add_weaver22_data(z2obs, file_read='2.0z2.5')

    # z3 obs
    z3obs = []
    in_redshift = np.where(zupMu13 == 3.0)
    z3obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), 'o'))
    #in_redshift = np.where(zdnS12 == 2.5)
    #z3obs.append((observation("Santini+2012", xobsS12[in_redshift], yobsS12[in_redshift], lerrS12[in_redshift], herrS12[in_redshift], err_absolute=False), 'o'))
    add_thorne21_data(z3obs, file_read='z3')
    add_weaver22_data(z3obs, file_read='3.0z3.5')

    # z4 obs
    z4obs = []
    in_redshift = np.where(zupMu13 == 4.0)
    z4obs.append((observation("Muzzin+2013", xobsMu13[in_redshift], yobsMu13[in_redshift], lerrMu13[in_redshift], herrMu13[in_redshift], err_absolute=False), 'o'))
    #in_redshift = np.where(zdnS12 == 3.5)
    #z4obs.append((observation("Santini+2012", xobsS12[in_redshift], yobsS12[in_redshift], lerrS12[in_redshift], herrS12[in_redshift], err_absolute=False), 'o'))
    add_thorne21_data(z4obs, file_read='z4')
    add_weaver22_data(z4obs, file_read='3.5z4.5')

    return (z0obs, z05obs, z1obs, z2obs, z3obs, z4obs)


def fill_arrays(h0, typeg, mbh, mstars, sfr, mh, mh1, BH, SFR, BHSFR, massgal, SMH1M, index, only_centrals, only_centrals_hmf):

    bin_it   = functools.partial(us.wmedians, xbins=xmf, nmin=10)
    
    # all branches (all)
    if only_centrals == 0:
        ind = np.where((mbh> 0)&(typeg==0))
    elif only_centrals == 1:
        ind = np.where((mbh> 0)&(typeg==1))
    elif only_centrals == 2:
        ind = np.where((mbh> 0)&(typeg==2))
    else:
        ind = np.where(mbh > 0)
    if include_indv == True:
        x1 = np.log10(mstars[ind]) - np.log10(float(h0))
        y1 = np.log10(mbh[ind]) - np.log10(float(h0))       
    BH[index,:] = bin_it(x=np.log10(mstars[ind]) - np.log10(float(h0)),
                    y=np.log10(mbh[ind]) - np.log10(float(h0)))

    if only_centrals == 0:
        ind = np.where((sfr>0)&(mstars> 0)&(typeg==0))
    elif only_centrals == 1:
        ind = np.where((sfr>0)&(mstars> 0)&(typeg==1))
    elif only_centrals == 2:
        ind = np.where((sfr>0)&(mstars> 0)&(typeg==2))
    else:
        ind = np.where((sfr>0)&(mstars> 0)) 
    if include_indv == True:
        x2 = np.log10(mstars[ind]) - np.log10(float(h0))
        y2 = np.log10(sfr[ind]) - np.log10(float(h0))
    SFR[index,:] = bin_it(x=np.log10(mstars[ind]) - np.log10(float(h0)),
                    y=np.log10(sfr[ind])- np.log10(float(h0)))
    
    ssfr = sfr / mstars
    #apply lower limit to ssfr
    ind = np.where(ssfr < 1e-14)
    ssfr[ind] = 1e-14
    mstars_l = 1e10
    if only_centrals == 0:
        ind = np.where((mbh > 0) & (mstars/h0 > mstars_l) & (typeg <= 0))
    elif only_centrals == 1:
        ind = np.where((mbh > 0) & (mstars/h0 > mstars_l) & (typeg == 1))
    elif only_centrals == 2:
        ind = np.where((mbh > 0) & (mstars/h0 > mstars_l) & (typeg == 2))
    else:
        ind = np.where((mbh > 0) & (mstars/h0 > mstars_l))
    if include_indv == True:
        x3 = np.log10(mbh[ind]) - np.log10(float(h0))
        y3 = np.log10(ssfr[ind])
    BHSFR[index,:] = bin_it(x=np.log10(mbh[ind]) - np.log10(float(h0)), 
                            y=np.log10(ssfr[ind]))
    if only_centrals_hmf == 0:
        ind = np.where((typeg <= 0) & (mstars > 0))
    elif only_centrals_hmf == 1:
        ind = np.where((typeg == 1) & (mstars > 0))
    elif only_centrals_hmf == 2:
        ind = np.where((typeg == 2) & (mstars > 0))
    else:
        ind = np.where(mstars > 0) 
    if include_indv == True:
        x4 = np.log10(mh[ind]) - np.log10(float(h0))
        y4 = np.log10(mstars[ind]) - np.log10(float(h0))
    massgal[index,:] = us.wmedians(x=np.log10(mh[ind]) - np.log10(float(h0)),
                                y=np.log10(mstars[ind]) - np.log10(float(h0)),
                                xbins=xmf)

    if only_centrals == 0:
        ind = np.where((typeg <= 0) & (mstars > 0) & (mh1 > 0))
    elif only_centrals == 1:
        ind = np.where((typeg == 1) & (mstars > 0) & (mh1 > 0))
    elif only_centrals == 2:
        ind = np.where((typeg == 2) & (mstars > 0) & (mh1 > 0))
    else:
        ind = np.where((mstars > 0) & (mh1 > 0))
    if include_indv == True:
        x5 = np.log10(mstars[ind]) - np.log10(float(h0))
        y5 = np.log10(mh1[ind]) - np.log10(float(h0))
    SMH1M[index,:] = us.wmedians(x=np.log10(mstars[ind]) - np.log10(float(h0)),
                                y=np.log10(mh1[ind]) - np.log10(float(h0)),
                                xbins=xmf)
    if include_indv == True:
        return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5
    else:
        return

def fill_arrays_vol(h0, volh, typeg, mstars, mh, HMF, SMF, index, only_centrals, only_centrals_hmf):

    bin_it   = functools.partial(us.wmedians, xbins=xmf, nmin=10)

    # all branches (all)
    if only_centrals_hmf == 0:
        ind = np.where((mh> 0)&(mstars>1e5)&(typeg==0))
    elif only_centrals_hmf == 1:
        ind = np.where((mh> 0)&(mstars>1e5)&(typeg==1))
    elif only_centrals_hmf == 2:
        ind = np.where((mh> 0)&(mstars>1e5)&(typeg==2))
    else:
        ind = np.where(mh > 0) 

    H, bins_edges = np.histogram(np.log10(mh[ind])-np.log10(float(h0)),bins=np.append(mbins,mupp))
    HMF[index,:] = HMF[index,:] + H
    vol = volh/pow(h0,3.)  # In Mpc^3
    HMF[index,:] = HMF[index,:]/vol/dm

    if only_centrals == 0:
        ind = np.where((mstars> 0)&(typeg==0))
    elif only_centrals == 1:
        ind = np.where((mstars> 0)&(typeg==1))
    elif only_centrals == 2:
        ind = np.where((mstars> 0)&(typeg==2))
    else:
        ind = np.where((mstars> 0)) 

    H, bins_edges = np.histogram(np.log10(mstars[ind])-np.log10(float(h0)),bins=np.append(mbins,mupp))
    SMF[index,:] = SMF[index,:] + H
    SMF[index,:] = SMF[index,:]/vol/dm
    
    return

def fill_arrays_csfrd(h0, volh, typeg, sfr, CSFRD, index, only_centrals):

    bin_it   = functools.partial(us.wmedians, xbins=xmf, nmin=10)

    # all branches (all)
    if only_centrals == 0:
        ind = np.where((sfr> 0)&(typeg==0))
    elif only_centrals == 1:
        ind = np.where((sfr> 0)&(typeg==1))
    elif only_centrals == 2:
        ind = np.where((sfr> 0)&(typeg==2))
    else:
        ind = np.where((sfr> 0)) 

    sfr_global = sum(sfr[ind]) / volh
    CSFRD[index] = sfr_global*pow(h0,2.0)

    return


def fill_arrays_sh(h0, typeg, msh, mstars, mh, mbh, fedd, mshH, eddBH, index, only_centrals):

    bin_it   = functools.partial(us.wmedians, xbins=xmf, nmin=10)

    # all branches (all)
    if only_centrals == 0:
        ind = np.where((mh> 0)&(msh> 0)&(typeg==0))
    elif only_centrals == 1:
        ind = np.where((mh> 0)&(msh> 0)&(typeg==1))
    elif only_centrals == 2:
        ind = np.where((mh> 0)&(msh> 0)&(typeg==2))
    else:
        ind = np.where((mh> 0)&(msh> 0))
    if include_indv == True:
        x5 = np.log10(mh[ind]) - np.log10(float(h0))
        y5 = np.log10(msh[ind]) - np.log10(msh[ind]+mstars[ind])
    mshH[index,:] = bin_it(x=np.log10(mh[ind]) - np.log10(float(h0)),
                    y=np.log10(msh[ind]) - np.log10(msh[ind]+mstars[ind]))

    # all branches (all)
    if only_centrals == 0:
        ind = np.where((mbh> 0)&(fedd> 0)&(typeg==0))
    elif only_centrals == 1:
        ind = np.where((mbh> 0)&(fedd> 0)&(typeg==1))
    elif only_centrals == 2:
        ind = np.where((mbh> 0)&(fedd> 0)&(typeg==2))
    else:
        ind = np.where((mbh> 0)&(fedd> 0))
    if include_indv == True:
        x6 = np.log10(mbh[ind]) - np.log10(float(h0))
        y6 = np.log10(fedd[ind]) - np.log10(float(h0))
    eddBH[index,:] = bin_it(x=np.log10(mbh[ind]) - np.log10(float(h0)),
                    y=np.log10(fedd[ind]) - np.log10(float(h0)))

    if include_indv == True:
        return x5, y5, x6, y6
    else:
        return


def fill_arrays_sfh(h0, typeg_sfh, mstars_sfh, sfh, LBT, SFH_med, only_centrals):

    ngals       = len(sfh[:,0])
    mbins = (9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5, 11.75, 12.5)
    for j in range(0,len(mbins)-1):
        if only_centrals == 0:
            ind = np.where((mstars_sfh > 10**mbins[j]) & (mstars_sfh < 10**mbins[j+1]) & (typeg_sfh == 0))
        elif only_centrals == 1:
            ind = np.where((mstars_sfh > 10**mbins[j]) & (mstars_sfh < 10**mbins[j+1]) & (typeg_sfh == 1))
        elif only_centrals == 2:
            ind = np.where((mstars_sfh > 10**mbins[j]) & (mstars_sfh < 10**mbins[j+1]) & (typeg_sfh == 2))
        else:
            ind = np.where((mstars_sfh > 10**mbins[j]) & (mstars_sfh < 10**mbins[j+1]))
        if(len(mstars_sfh[ind]) > 0):
            tot_sfh_selec = sfh[ind,:]
            tot_sfh_selec = tot_sfh_selec[0,:]
            if(ngals >= 10):
                print("Stellar mass bin", mbins[j], " has ", ngals, " galaxies") 
                for snap in range(0,len(LBT)):
                    SFH_med[0,snap,j] = np.median(tot_sfh_selec[:,snap])
                    SFH_med[1,snap,j] = np.percentile(tot_sfh_selec[:,snap],16)
                    SFH_med[2,snap,j] = np.percentile(tot_sfh_selec[:,snap],84)
                    if(SFH_med[0,snap,j] < 0.001):
                        SFH_med[0,snap,j] = 0.001
                        SFH_med[1,snap,j] = 0.001 - 0.001*0.2
                        SFH_med[2,snap,j] = 0.001 + 0.001*0.2

    return


def plot_mstars_BH(plt, outdir, BH_a, BH_ms, BH_mt2, BH_art, BH_nart, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    #stellar mass-black hole mass relation
    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\star}/M_{\odot})$"
    ytit = "$\\rm log_{10} (\\rm M_{\\rm BH}/M_{\odot})$"

#    xmin, xmax, ymin, ymax = 0, 13, 3, 11
    xmin, xmax, ymin, ymax = 9, 13, 6, 11
    xleg = xmax - 0.2 * (xmax - xmin)
    yleg = ymin + 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    #stellar mass-black hole mass relation
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(BH_a[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = BH_a[iz,0,ind]
    errdn = BH_a[iz,1,ind]
    errup = BH_a[iz,2,ind]
    ax.plot(xplot,yplot[0],color='k',label="Shark ("+lab+")",linewidth=linw1)
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey',alpha=0.25, interpolate=True)

    ind = np.where(BH_art[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = BH_art[iz,0,ind]
    errdn = BH_art[iz,1,ind]
    errup = BH_art[iz,2,ind]
    ax.plot(xplot,yplot[0],color='Navy',label="Shark ("+lab+" art)",linewidth=linw1)
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy',alpha=0.25, interpolate=True)

    ind = np.where(BH_nart[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = BH_nart[iz,0,ind]
    errdn = BH_nart[iz,1,ind]
    errup = BH_nart[iz,2,ind]
    ax.plot(xplot,yplot[0],color='green',label="Shark ("+lab+" no art)",linewidth=linw1)
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green',alpha=0.25, interpolate=True)

    if include_obs == True and iz==0:
        #BH-stellar mass relation
        ms, sfr, upperlimflag, mbh, mbherr = common.load_observation(obsdir, 'BHs/MBH_host_gals_Terrazas17.dat', [0,1,2,3,4])
        ind=np.where(sfr-ms > -11.5)
        ax.errorbar(ms[ind], mbh[ind], yerr=mbherr[ind], xerr=0.2, ls='None', mfc='None', ecolor = 'PowderBlue', mec='PowderBlue',marker='s', label="T17 (SF)")
        ind=np.where(sfr-ms <= -11.5)
        ax.errorbar(ms[ind], mbh[ind], yerr=mbherr[ind], xerr=0.2, ls='None', mfc='None', ecolor = 'LightSalmon', mec='LightSalmon',marker='s',label="T17 (P)")
        
        common.prepare_legend(ax, ['k','navy','green','PowderBlue','LightSalmon'], loc=2)
    else:
        common.prepare_legend(ax, ['k','navy','green'], loc=2)
        
    common.savefig(outdir, fig, 'stellarmass-BH_'+str(z)+'.png')
    plt.close()

                            
def plot_mstars_SFR(plt, outdir, h0, BH_a, BH_ms, BH_nms, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    #stellar mass-black hole mass relation
    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\star}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm SFR/M_{\odot} yr^{-1})$"

    #xmin, xmax, ymin, ymax = 0, 13, -12, 4
    xmin, xmax, ymin, ymax = 8, 13, -4, 4
    xleg = xmax - 0.2 * (xmax - xmin)
    yleg = ymin + 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    #stellar mass-black hole mass relation
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(BH_a[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = BH_a[iz,0,ind]
    errdn = BH_a[iz,1,ind]
    errup = BH_a[iz,2,ind]
    ax.plot(xplot,yplot[0],color='k',label="Shark ("+lab+")",linewidth=linw1)
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey',alpha=0.25, interpolate=True)

    ind = np.where(BH_ms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = BH_ms[iz,0,ind]
    errdn = BH_ms[iz,1,ind]
    errup = BH_ms[iz,2,ind]
    ax.plot(xplot,yplot[0],color='Navy',label="Shark ("+lab+" art)",linewidth=linw1)
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy',alpha=0.25, interpolate=True)

    ind = np.where(BH_nms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = BH_nms[iz,0,ind]
    errdn = BH_nms[iz,1,ind]
    errup = BH_nms[iz,2,ind]
    ax.plot(xplot,yplot[0],color='green',label="Shark ("+lab+" no art)",linewidth=linw1)
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green',alpha=0.25, interpolate=True)

    if include_obs == True and iz==0:
        #SFR relation z=0
        lm, SFR = common.load_observation(obsdir, 'SFR/Brinchmann04.dat', (0, 1))
        hobs = 0.7
        #add cosmology correction plus IMF correction that goes into the stellar mass.
        corr_cos = np.log10(pow(hobs,2)/pow(h0,2)) - 0.09
        #apply correction to both stellar mass and SFRs.
        ax.plot(lm[0:35] + corr_cos, SFR[0:35] + corr_cos, color='SandyBrown', linewidth = 4, linestyle='dashed', label='Brinchmann+04')
        #ax.plot(lm[36:70] + corr_cos, SFR[36:70] + corr_cos, color='PaleVioletRed',linewidth = 5, linestyle='dotted')
        #ax.plot(lm[71:len(SFR)] + corr_cos, SFR[71:len(SFR)] + corr_cos, color='PaleVioletRed',linewidth = 5, linestyle='dotted')

        xdataD16 = [9.3, 10.6]
        ydataD16 = [-0.39, 0.477]
        ax.plot(xdataD16,ydataD16, color='Crimson',linestyle='dashdot',linewidth = 4, label='Davies+16')

        #GAMA data at z<0.06
        #CATAID StellarMass_bestfit StellarMass_50 StellarMass_16 StellarMass_84 SFR_bestfit SFR_50 SFR_16 SFR_84 Zgas_bestfit Zgas_50 Zgas_16 Zgas_84 DustMass_bestfit DustMass_50 DustMass_16 DustMass_84 DustLum_50\DustLum_16 DustLum_84 uberID redshift
        ms_gama, sfr_gama = common.load_observation(obsdir, 'GAMA/ProSpect_Claudia.txt', [2,6])
        ind = np.where(sfr_gama < 1e-3)
        sfr_gama[ind] = 1e-3
        #ax.hexbin(np.log10(ms_gama), np.log10(sfr_gama), gridsize=(20,20), mincnt=5) #, cmap = 'plasma') #, **contour_kwargs)
        us.density_contour_reduced(ax, np.log10(ms_gama), np.log10(sfr_gama), 25, 25) #, **contour_kwargs)

        bin_it   = functools.partial(us.wmedians, xbins=xmf, nmin=10)
        toplot = bin_it(x=np.log10(ms_gama), y=np.log10(sfr_gama))
        ind = np.where(toplot[0,:] != 0)
        yp = toplot[0,ind]
        yup = toplot[2,ind]
        ydn = toplot[1,ind]
        ax.plot(xmf[ind], yp[0],color='Maroon',linestyle='dashed', linewidth = 5, label="Bellstedt+20")
        #ax.plot(xmf[ind], yp[0]+yup[0],color='PaleVioletRed',linestyle='dotted', linewidth = 5)
        #ax.plot(xmf[ind], yp[0]-ydn[0],color='PaleVioletRed',linestyle='dotted', linewidth = 5)
        
        # individual massive galaxies from Terrazas+17
        ms, sfr, upperlimflag = common.load_observation(obsdir, 'BHs/MBH_host_gals_Terrazas17.dat', [0,1,2])
        ind = np.where(ms > 11.3)
        ax.errorbar(ms[ind], sfr[ind], xerr=0.2, yerr=0.3, ls='None', mfc='None', ecolor = 'r', mec='r',marker='s',label="Terrazas+17")
        ind = np.where((upperlimflag == 1) & (ms > 11.3))
        for a,b in zip (ms[ind], sfr[ind]):
            ax.arrow(a, b, 0, -0.3, head_width=0.05, head_length=0.1, fc='r', ec='r')        
        common.prepare_legend(ax, ['k','navy','green','SandyBrown','Crimson','Maroon','r'], loc=2)
    else:
        common.prepare_legend(ax, ['k','navy','green'], loc=2)

    common.savefig(outdir, fig, 'stellarmass-SFR_'+str(z)+'.png')
    plt.close()

def plot_BH_SSFR(plt, outdir, BH_a, BH_ms, BH_nms, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    #SSFR vs BH mass
    fig = plt.figure(figsize=(6,5.5))
    ytit = "$\\rm log_{10} (\\rm sSFR/M_{\odot} yr^{-1})$"
    xtit = "$\\rm log_{10} (\\rm M_{\\rm BH}/M_{\odot})$"

    #xmin, xmax, ymin, ymax = 3, 11, -14.5, -7
    xmin, xmax, ymin, ymax = 5, 11, -14, -7
    xleg = xmax - 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)
    ax.text(xleg-0.25, yleg-0.3, '$M_{\\star}> 10^{10}\\, M_{\\odot}$', fontsize=12)

    #xL18, yL18, yl_L18, yu_L18 = common.load_observation(obsdir, 'Models/SharkVariations/BHSSFR_Lagos18.dat', [0,1,2,3])
    #ax.plot(xL18, yL18,color='k',label="Shark v1.1 (L18)")
    #ax.fill_between(xL18,yl_L18,yu_L18, facecolor='k', alpha=0.25, interpolate=True)

    #Predicted BH-bulge mass relation
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(BH_a[iz,0,:] != 0)
    if(len(xmf[ind]) > 0):
        xplot = xmf[ind]
        yplot = BH_a[iz,0,ind]
        errdn = BH_a[iz,1,ind]
        errup = BH_a[iz,2,ind]
        ax.plot(xplot,yplot[0],color='k',label="Shark ("+lab+")",linewidth=linw1)
        ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey', alpha=0.25, interpolate=True)

    ind = np.where(BH_ms[iz,0,:] != 0)
    if(len(xmf[ind]) > 0):
        xplot = xmf[ind]
        yplot = BH_ms[iz,0,ind]
        errdn = BH_ms[iz,1,ind]
        errup = BH_ms[iz,2,ind]
        ax.plot(xplot,yplot[0],color='Navy',label="Shark ("+lab+" art)",linewidth=linw1)
        ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy', alpha=0.25, interpolate=True)

    ind = np.where(BH_nms[iz,0,:] != 0)
    if(len(xmf[ind]) > 0):
        xplot = xmf[ind]
        yplot = BH_nms[iz,0,ind]
        errdn = BH_nms[iz,1,ind]
        errup = BH_nms[iz,2,ind]
        ax.plot(xplot,yplot[0],color='green',label="Shark ("+lab+" no art)",linewidth=linw1)
        ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green', alpha=0.25, interpolate=True)

    if include_obs == True and iz==0:
        #BH-SSFR relation
        ms, sfr, upperlimflag, mbh, mbherr = common.load_observation(obsdir, 'BHs/MBH_host_gals_Terrazas17.dat', [0,1,2,3,4])
        ax.errorbar(mbh, sfr-ms, xerr=mbherr, yerr=0.3, ls='None', mfc='None', ecolor = 'b', mec='b',marker='s',label="Terrazas+17")
        ind = np.where(upperlimflag == 1)
        for a,b in zip (mbh[ind], sfr[ind]-ms[ind]):
            ax.arrow(a, b, 0, -0.3, head_width=0.05, head_length=0.1, fc='b', ec='b')
        common.prepare_legend(ax, ['k','navy','green','b'], loc=2)
    else:
        common.prepare_legend(ax, ['k','navy','green'], loc=2)

    common.savefig(outdir, fig, 'BH-sSFR_'+str(z)+'.png')
    plt.close()

def plot_HM_SM(plt, outdir, massgal_a, massgal_ms, massgal_nms, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    def plot_moster13(ax, z, label):
        #Moster et al. (2013) abundance matching SMHM relation
        M10 = 11.590
        M11 = 1.195
        N10 = 0.0351
        N11 = -0.0247
        beta10 = 1.376
        beta11 = -0.826
        gamma10 = 0.608
        gamma11 = 0.329
        M1 = pow(10.0, M10 + M11 * z/(z+1))
        N = N10 + N11 * z/(z+1)
        beta = beta10 + beta11 * z/(z+1)
        gamma = gamma10 + gamma11 * z/(z+1)

        mh = pow(10.0,xmf)
        m = mh * 2*N * pow (( pow(mh/M1, -beta ) + pow(mh/M1, gamma)), -1)

        ax.plot(xmf,np.log10(m),'r', linestyle='dashed', color='r', linewidth=3, label=label)

    def plot_berhoozi13(ax, z, label):
        a = 1.0/(1.0+z)
        nu = np.exp(-4*a*a)
        log_epsilon = -1.777 + (-0.006*(a-1)) * nu
        M1= 11.514 + ( - 1.793 * (a-1) - 0.251 * z) * nu
        alpha = -1.412 + 0.731 * nu * (a-1)
        delta = 3.508 + (2.608*(a-1)-0.043 * z) * nu
        gamma = 0.316 + (1.319*(a-1)+0.279 * z) * nu
        Min = xmf-M1
        fx = -np.log10(pow(10,alpha*Min)+1.0)+ delta * pow(np.log10(1+np.exp(Min)),gamma) / (1+np.exp(pow(10,-Min)))
        f = -0.3+ delta * pow(np.log10(2.0),gamma) / (1+np.exp(1))

        m = log_epsilon + M1 + fx - f

        ax.plot(xmf,m, 'b', linestyle='dashdot',color='b',linewidth=3, label=label)


    def plot_observations_kravtsov18(ax):

        mh, sm = common.load_observation(obsdir, 'SMHM/SatKinsAndClusters_Kravtsov18.dat', [0,1])
        ax.errorbar(mh, sm, xerr=0.2, yerr=0.2, color='purple', marker='s',  ls='None', label='Kravtsov+18')
    
    def plot_observations_romeo20(ax):

        mh, smmh = common.load_observation(obsdir, 'SMHM/Romeo20_SMHM.dat', [0,1])
        sm = np.log10(10**smmh * 10**mh)
        ax.plot(mh, sm, color='orange', marker='*',  ls='None', label='Romeo+20', fillstyle='none')

    def plot_observations_taylor20(ax):
        sm, sml, smh, hm, hml, hmh = common.load_observation(obsdir, 'SMHM/Taylor20.dat', [0,1,2,3,4,5])
        ax.errorbar(np.log10(hm*1e12), sm, xerr=[np.log10(hm)-np.log10(hml), np.log10(hmh)-np.log10(hm)], yerr=[sm-sml, smh-sm], color='Salmon', marker='d',  ls='None', label='Taylor+20')

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\rm halo}/M_{\odot})$"
    ytit = "$\\rm log_{10} (\\rm M_{\\star}/M_{\odot})$"
    #xmin, xmax, ymin, ymax = 8, 15, 0, 13
    xmin, xmax, ymin, ymax = 10, 15, 7, 13
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    #Predicted SMHM
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(massgal_a[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_a[iz,0,ind]
    errdn = massgal_a[iz,1,ind]
    errup = massgal_a[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_ms[iz,0,ind]
    errdn = massgal_ms[iz,1,ind]
    errup = massgal_ms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_nms[iz,0,ind]
    errdn = massgal_nms[iz,1,ind]
    errup = massgal_nms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='green', label="Shark ("+lab+" no art)",linewidth=linw1)
    
    if include_obs == True:
        if iz == 0:
            plot_moster13(ax, z, 'Moster+13')
            plot_berhoozi13(ax, z, 'Behroozi+13')
            plot_observations_kravtsov18(ax)
            plot_observations_romeo20(ax)
            plot_observations_taylor20(ax)
            common.prepare_legend(ax, ['r','b','k','navy','green','purple','orange','Salmon'], loc=4)
        else:
            plot_moster13(ax, z, 'Moster+13')
            plot_berhoozi13(ax, z, 'Behroozi+13')
            common.prepare_legend(ax, ['r','b','k','navy','green'], loc=4)

    common.savefig(outdir, fig, 'HM-SM_'+str(z)+'.png')
    plt.close()


def plot_SM_H1M(plt, outdir, massgal_a, massgal_ms, massgal_nms, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\rm stars}/M_{\odot})$"
    ytit = "$\\rm log_{10} (\\rm M_{\\rm H1}/M_{\odot})$"
    #xmin, xmax, ymin, ymax = 0, 13, 0, 13
    xmin, xmax, ymin, ymax = 7, 13, 6, 13
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    #Predicted SMHM
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(massgal_a[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_a[iz,0,ind]
    errdn = massgal_a[iz,1,ind]
    errup = massgal_a[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_ms[iz,0,ind]
    errdn = massgal_ms[iz,1,ind]
    errup = massgal_ms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_nms[iz,0,ind]
    errdn = massgal_nms[iz,1,ind]
    errup = massgal_nms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='green', label="Shark ("+lab+" no art)",linewidth=linw1)
    
    common.prepare_legend(ax, ['k','navy','green'], loc=4)
    common.savefig(outdir, fig, 'SM-H1M_'+str(z)+'.png')
    plt.close()


def plot_HMF(plt, outdir, h0, omega_m, massgal_a, massgal_ms, massgal_nms, z, iz, only_centrals, all_branches):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\rm halo}/M_{\odot})$"
    ytit = "$\\rm log_{10}(\Phi/dlog{\\rm M_{\\rm halo}}/{\\rm Mpc}^{-3} )$"
    #xmin, xmax, ymin, ymax = 8, 15, -4, -1
    xmin, xmax, ymin, ymax = 10, 16, -7, -1
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    ind = np.where(massgal_a[iz,:] != 0)
    xplot = xmf[ind]
    yplot = np.log10(massgal_a[iz,ind])
    ax.plot(xplot, yplot[0], color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms[iz,:] != 0)
    xplot = xmf[ind]
    yplot = np.log10(massgal_ms[iz,ind])
    ax.plot(xplot, yplot[0], color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nms[iz,:] != 0)
    xplot = xmf[ind]
    yplot = np.log10(massgal_nms[iz,ind])
    ax.plot(xplot, yplot[0], color='green', label="Shark ("+lab+" no art)",linewidth=linw1)

    if include_obs == True:
        if iz==0:
            #HMF calc HMF calculated by Sheth & Tormen (2001)
            lmp, dp = common.load_observation(obsdir, 'mf/HMF/mVector_PLANCK-SMT_z0.dat', [0, 7])
            lmp_plot = np.log10(lmp) - np.log10(h0)
            dp_plot = np.log10(dp) + np.log10(pow(h0,3.))
            ax.plot(lmp_plot,dp_plot,'r--', label = 'HMF calc',lw=3)
#            lm, p, dpdn, dpup = common.load_observation(obsdir, 'mf/SMF/GAMAII_BBD_GSMFs.dat', [0,1,2,3])
#            xobs = lm
#            indx = np.where(p > 0)
#            yobs = np.log10(p[indx])
#            ydn = yobs - np.log10(p[indx]-dpdn[indx])
#            yup = np.log10(p[indx]+dpup[indx]) - yobs
#            ax.errorbar(xobs[indx], yobs, ydn, yup, 'ro', label='Wright+17')
            common.prepare_legend(ax, ['k','navy','green','r'], loc=4)
        elif iz==1:
            #HMF calc HMF calculated by Sheth & Tormen (2001)
            lmp, dp = common.load_observation(obsdir, 'mf/HMF/mVector_PLANCK-SMT_z05.dat', [0, 7])
            lmp_plot = np.log10(lmp) - np.log10(h0)
            dp_plot = np.log10(dp) + np.log10(pow(h0,3.))
            ax.plot(lmp_plot,dp_plot,'r--', label = 'HMF calc',lw=3)
            common.prepare_legend(ax, ['k','navy','green','r'], loc=4)
        elif iz==2:
            #HMF calc HMF calculated by Sheth & Tormen (2001)
            lmp, dp = common.load_observation(obsdir, 'mf/HMF/mVector_PLANCK-SMT_z1.dat', [0, 7])
            lmp_plot = np.log10(lmp) - np.log10(h0)
            dp_plot = np.log10(dp) + np.log10(pow(h0,3.))
            ax.plot(lmp_plot,dp_plot,'r--', label = 'HMF calc',lw=3)
            common.prepare_legend(ax, ['k','navy','green','r'], loc=4)
        elif iz==3:
            #HMF calc HMF calculated by Sheth & Tormen (2001)
            lmp, dp = common.load_observation(obsdir, 'mf/HMF/mVector_PLANCK-SMT_z2.dat', [0, 7])
            lmp_plot = np.log10(lmp) - np.log10(h0)
            dp_plot = np.log10(dp) + np.log10(pow(h0,3.))
            ax.plot(lmp_plot,dp_plot,'r--', label = 'HMF calc',lw=3)
            common.prepare_legend(ax, ['k','navy','green','r'], loc=4)
        else:
            common.prepare_legend(ax, ['k','navy','green'], loc=4)
    
    common.savefig(outdir, fig, 'HMF_'+str(z)+'.png')
    plt.close()


def plot_SMF(plt, outdir, h0, massgal_a, massgal_ms, massgal_nms, z, iz, only_centrals, all_branches):

    (z0obs, z05obs, z1obs, z2obs, z3obs, z4obs) = load_smf_observations(obsdir, h0)

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\star}/M_{\odot})$"
    ytit = "$\\rm log_{10}(\Phi/dlog_{10}{\\rm M_{\\star}}/{\\rm Mpc}^{-3} )$"
    #xmin, xmax, ymin, ymax = 5, 12, -4, -1
    xmin, xmax, ymin, ymax = 8, 13, -6, -1
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    ind = np.where(massgal_a[iz,:] != 0)
    xplot = xmf[ind]
    yplot = np.log10(massgal_a[iz,ind])
    ax.plot(xplot, yplot[0], color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms[iz,:] != 0)
    xplot = xmf[ind]
    yplot = np.log10(massgal_ms[iz,ind])
    ax.plot(xplot, yplot[0], color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nms[iz,:] != 0)
    xplot = xmf[ind]
    yplot = np.log10(massgal_nms[iz,ind])
    ax.plot(xplot, yplot[0], color='green', label="Shark ("+lab+" no art)",linewidth=linw1)

    observations = (z0obs, z05obs, z1obs, z2obs, z3obs, z4obs)
    if include_obs == True:
            # Observations
            if iz==0:
                for obs, marker in z0obs:
                    common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                                marker, err_absolute=obs.err_absolute, label=obs.label, markersize=4)
                common.prepare_legend(ax, ['k','navy','green','grey','grey','grey'], loc=4)
            elif iz==1:
                for obs, marker in z05obs:
                    common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                                marker, err_absolute=obs.err_absolute, label=obs.label, markersize=4)
                common.prepare_legend(ax, ['k','navy','green','grey','grey','grey'], loc=4)
            elif iz==2:
                for obs, marker in z1obs:
                    common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                                marker, err_absolute=obs.err_absolute, label=obs.label, markersize=4)
                common.prepare_legend(ax, ['k','navy','green','grey','grey','grey'], loc=4)
            elif iz==3:
                for obs, marker in z2obs:
                    common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                                marker, err_absolute=obs.err_absolute, label=obs.label, markersize=4)
                common.prepare_legend(ax, ['k','navy','green','grey','grey','grey'], loc=4)
            elif iz==4:
                for obs, marker in z3obs:
                    common.errorbars(ax, obs.x, obs.y, obs.yerrdn, obs.yerrup, 'grey',
                                marker, err_absolute=obs.err_absolute, label=obs.label, markersize=4)
                common.prepare_legend(ax, ['k','navy','green','grey','grey','grey'], loc=4)
            else:
                common.prepare_legend(ax, ['k','navy','green'], loc=4)

    common.savefig(outdir, fig, 'SMF_'+str(z)+'.png')
    plt.close()


def plot_CSFRD(plt, outdir, h0, z_out, massgal_a, massgal_ms, massgal_mt2, massgal_art, massgal_nart, only_centrals, all_branches):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm redshift$"
    ytit = "$\\rm log_{10}(CSFRD/ M_{\odot}\,yr^{-1}\,cMpc^{-3})$"
    #xmin, xmax, ymin, ymax = 0, 10, -4, -0.5
    xmin, xmax, ymin, ymax = 0, 15, -6, -0.5
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))

    ind = np.where(massgal_a != 0)
    xplot = z_out[ind]
    yplot = np.log10(massgal_a[ind])
    ax.plot(xplot, yplot, color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms != 0)
    xplot = z_out[ind]
    yplot = np.log10(massgal_ms[ind])
    ax.plot(xplot, yplot, color='pink', label="Shark ("+lab+" MS)",linewidth=linw2)

    ind = np.where(massgal_mt2 != 0)
    xplot = z_out[ind]
    yplot = np.log10(massgal_mt2[ind])
    ax.plot(xplot, yplot, color='purple', label="Shark ("+lab+" MT)",linewidth=linw2)

    ind = np.where(massgal_art != 0)
    xplot = z_out[ind]
    yplot = np.log10(massgal_art[ind])
    ax.plot(xplot, yplot, color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nart != 0)
    xplot = z_out[ind]
    yplot = np.log10(massgal_nart[ind])
    ax.plot(xplot, yplot, color='green', label="Shark ("+lab+" no art)",linewidth=linw1)

    if include_obs == True:
        #Madau & Dickinson 2014
        reddnM14, redupM14, sfrM14, sfrM14errup, sfrM14errdn = common.load_observation(obsdir, 'Global/SFRD_Madau14.dat', [0,1,2,3,4])
        #authors assume a Salpeter IMF, so a correction of np.log10(0.63) is necessary.
        sfrM14errdn = abs(sfrM14errdn)
        hobs = 0.7
        sfrM14 = sfrM14 + np.log10(pow(hobs/h0, 2.0)) + np.log10(0.63)
        
        #D'Silva+23 (Chabrier IMF)
        sfrD23, sfrD23errup, sfrD23errdn, zD23, zD23errup, zD23errdn = common.load_observation(obsdir, 'Global/DSilva23_sfr.dat', [0,1,2,3,4,5])

        #Adams23 (Chabrier IMF)
        zA23, sfrA23, sfrA23errdn, sfrA23errup = common.load_observation(obsdir, 'Global/Adams23_CSFRDCompilation.dat', [0,1,2,3])
        sfrA23errdn = sfrA23 - sfrA23errdn #make them relative errors
        sfrA23errup = sfrA23errup - sfrA23
        hobs = 0.7
        yobsA23 = sfrA23 + np.log10(hobs/h0)
        
        hobs = 0.7
        yobsD23 = sfrD23 + np.log10(hobs/h0)

        #Baldry (Chabrier IMF), ['Baldry+2012, z<0.06']
        reddnM14, redupM14, sfrM14, sfrM14errup, sfrM14errdn = common.load_observation(obsdir, 'Global/SFRD_Madau14.dat', [0,1,2,3,4])
        #authors assume a Salpeter IMF, so a correction of np.log10(0.63) is necessary.
        sfrM14errdn = abs(sfrM14errdn)
        hobs = 0.7
        sfrM14 = sfrM14 + np.log10(pow(hobs/h0, 2.0)) + np.log10(0.63)
        #ax.errorbar((reddnM14 + redupM14) / 2.0, sfrM14, xerr=abs(redupM14-reddnM14)/2.0, yerr=[sfrM14errdn, sfrM14errup], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='D', markersize=1.5, label='Madau+14')

        ax.errorbar(zD23, yobsD23, xerr=[zD23errdn, zD23errup], yerr=[sfrD23errdn, sfrD23errup], ls='None', mfc='None', ecolor = 'red', mec='red',marker='o', label ='D\'Silva+23')
        ax.errorbar(zA23, yobsA23, yerr=[sfrA23errdn, sfrA23errup], ls='None', mfc='None', ecolor = 'darkgreen', mec='darkgreen',marker='s', label = 'Adams+23')
        #Driver (Chabrier IMF)
        redD17d, redD17u, sfrD17, err1, err2, err3, err4 = common.load_observation(obsdir, 'Global/Driver18_sfr.dat', [0,1,2,3,4,5,6])
        hobs = 0.7
        xobsD17 = (redD17d+redD17u)/2.0
        yobsD17 = sfrD17 + np.log10(hobs/h0)
        errD17 = yobsD17*0. - 999.
        errD17 = np.sqrt(pow(err1,2.0)+pow(err2,2.0)+pow(err3,2.0)+pow(err4,2.0))
        ax.errorbar(xobsD17, yobsD17, yerr=[errD17,errD17], ls='None', mfc='None', ecolor = 'darkorange', mec='darkorange',marker='o', label = 'Driver+18')

        common.prepare_legend(ax, ['k','pink','purple','navy','green','red','darkgreen','darkorange'], loc=3)
    else:
        common.prepare_legend(ax, ['k','pink','purple','navy','green'], loc=3)

    common.savefig(outdir, fig, 'CSFRD.png')
    plt.close()

    
def plot_HM_SHM(plt, outdir, massgal_a, massgal_ms, massgal_nms, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\rm halo}/M_{\odot})$"
    ytit = "$\\rm log_{10} (\\rm f_{\\rm IHL})$"
    xmin, xmax, ymin, ymax = 8, 15, -6, 0
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    #Predicted SMHM
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(massgal_a[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_a[iz,0,ind]
    errdn = massgal_a[iz,1,ind]
    errup = massgal_a[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_ms[iz,0,ind]
    errdn = massgal_ms[iz,1,ind]
    errup = massgal_ms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_nms[iz,0,ind]
    errdn = massgal_nms[iz,1,ind]
    errup = massgal_nms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='green', label="Shark ("+lab+" no art)",linewidth=linw1)
    
    common.prepare_legend(ax, ['k','navy','green'], loc=4)
    common.savefig(outdir, fig, 'HM-SHM_'+str(z)+'.png')
    plt.close()


def plot_BH_EDD(plt, outdir, massgal_a, massgal_ms, massgal_nms, z, iz, only_centrals, all_branches, x_ms=None,y_ms=None,x_mt2=None,y_mt2=None,x_nart=None,y_nart=None):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    fig = plt.figure(figsize=(6,5.5))
    xtit = "$\\rm log_{10} (\\rm M_{\\rm BH}/M_{\odot})$"
    ytit = "$\\rm log_{10} (\\rm f_{\\rm Edd})$"
    xmin, xmax, ymin, ymax = 4, 11, -40, 0
    xleg = xmin + 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1))
    ax.text(xleg, yleg, 'z='+str(z), fontsize=12)

    #Predicted SMHM
    if include_indv == True:
        ax.scatter(x_ms,y_ms,s=1,c='pink')
        ax.scatter(x_mt2,y_mt2,s=1,c='purple')
        ax.scatter(x_nart,y_nart,s=1,c='green')

    ind = np.where(massgal_a[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_a[iz,0,ind]
    errdn = massgal_a[iz,1,ind]
    errup = massgal_a[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='grey', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='k', label="Shark ("+lab+")",linewidth=linw1)

    ind = np.where(massgal_ms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_ms[iz,0,ind]
    errdn = massgal_ms[iz,1,ind]
    errup = massgal_ms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='Navy', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='Navy', label="Shark ("+lab+" art)",linewidth=linw1)

    ind = np.where(massgal_nms[iz,0,:] != 0)
    xplot = xmf[ind]
    yplot = massgal_nms[iz,0,ind]
    errdn = massgal_nms[iz,1,ind]
    errup = massgal_nms[iz,2,ind]
    ax.fill_between(xplot,yplot[0]+errup[0],yplot[0]-errdn[0], facecolor='green', alpha=0.25,interpolate=True)
    ax.errorbar(xplot, yplot[0], color='green', label="Shark ("+lab+" no art)",linewidth=linw1)
    
    common.prepare_legend(ax, ['k','navy','green'], loc=4)
    common.savefig(outdir, fig, 'BH-EDD_'+str(z)+'.png')
    plt.close()

    
def plot_individual_seds(plt, outdir, LBT, SFH_med_a, SFH_med_ms, SFH_med_mt, SFH_med_art, SFH_med_nart, ibin, only_centrals, all_branches):

    if all_branches == True:
        if only_centrals==0:
            lab = 'All Central'
        elif only_centrals==1:
            lab = 'All Type 1'
        elif only_centrals==2:
            lab = 'All Type 2'
        else:
            lab = 'All'
    else:
        if only_centrals==0:
            lab = 'MB Central'
        elif only_centrals==1:
            lab = 'MB Type 1'
        elif only_centrals==2:
            lab = 'MB Type 2'
        else:
            lab = 'MB'

    xtit="$\\rm LBT/Gyr$"
    ytit="$\\rm log_{10}(SFR/M_{\odot} yr^{-1})$"

    xmin = 0
    xmax, ymin, ymax = 13.6, -3, 3
    xleg = xmax + 0.025 * (xmax-xmin)
    yleg = ymax - 0.07 * (ymax-ymin)

    fig = plt.figure(figsize=(6.5,5))
    mbins = (9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5, 11.75, 12.5)
    colors = ('Navy','Blue','RoyalBlue','SkyBlue','Teal','DarkTurquoise','Aquamarine','Yellow', 'Gold',  'Orange','OrangeRed', 'LightSalmon', 'Crimson', 'Red', 'DarkRed')

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(2, 2, 1, 1))
    ax.text(xleg,yleg,'$\\rm log_{10}(M_{\\star}/M_{\\odot})$', fontsize=12)
    ax.text(0.5,2.8,'Shark v2.0 ('+lab+')', fontsize=13)

    ax.fill_between(LBT,np.log10(SFH_med_a[1,:,ibin]),np.log10(SFH_med_a[2,:,ibin]), facecolor='grey', alpha=0.5, interpolate=True)
    ax.plot(LBT,np.log10(SFH_med_a[0,:,ibin]), color='k', linewidth=linw1, label=' %s' % str(mbins[ibin]+0.125))

    ax.plot(LBT,np.log10(SFH_med_ms[0,:,ibin]), color='pink', linewidth=linw2, label=' MS %s' % str(mbins[ibin]+0.125))

    ax.plot(LBT,np.log10(SFH_med_mt[0,:,ibin]), color='purple', linewidth=linw2, label=' MT %s' % str(mbins[ibin]+0.125))

    ax.fill_between(LBT,np.log10(SFH_med_art[1,:,ibin]),np.log10(SFH_med_art[2,:,ibin]), facecolor='Navy', alpha=0.5, interpolate=True)
    ax.plot(LBT,np.log10(SFH_med_art[0,:,ibin]), color='Navy', linewidth=linw1, label=' art %s' % str(mbins[ibin]+0.125))

    ax.fill_between(LBT,np.log10(SFH_med_nart[1,:,ibin]),np.log10(SFH_med_nart[2,:,ibin]), facecolor='green', alpha=0.5, interpolate=True)
    ax.plot(LBT,np.log10(SFH_med_nart[0,:,ibin]), color='green', linewidth=linw1, label=' nart %s' % str(mbins[ibin]+0.125))

    common.prepare_legend(ax, ['k','pink','purple','Navy','green'], bbox_to_anchor=(0.98, 0.1))
    plt.tight_layout()
    common.savefig(outdir, fig, "SFHs_mbin"+str(mbins[ibin])+".png")


def main():

    sim = sys.argv[1]
    sim_dict = choose_sim(sim)
    omega_m = sim_dict['omega_m0']
    
    plot_dir = '/fred/oz009/achandro/track_halos/'+sim_dict['sim']+'/'
    outdir = '/fred/oz009/achandro/shark_output/Plots/'+sim_dict['sim']+'/'+shark_model+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if all_branches == True:

        # ALL BRANCHES
        # mask files
        node_mask = np.array([],dtype='int64')
        mask_ms = np.array([],dtype='int32')
        mask_mt2 = np.array([],dtype='int32')

        for subv in range(sim_dict['dhalo_subvol']):
            if (sim_dict['snapshots']-1)<100:
                if os.path.isfile(plot_dir+'mask_tree_0'+str(sim_dict['snapshots']-1)+"."+str(subv)+".hdf5")==True:
                    f = h5py.File(plot_dir+'mask_tree_0'+str(sim_dict['snapshots']-1)+"."+str(subv)+".hdf5",'r')
            else:
                if os.path.isfile(plot_dir+'mask_tree_0'+str(sim_dict['snapshots']-1)+"."+str(subv)+".hdf5")==True:
                    f = h5py.File(plot_dir+'mask_tree_'+str(sim_dict['snapshots']-1)+"."+str(subv)+".hdf5",'r')
            node_mask = np.concatenate([node_mask,f["nodeIndex"][()]])
            mask_ms = np.concatenate([mask_ms,f["ms"][()]])
            mask_mt2 = np.concatenate([mask_mt2,f["mt3sigma_z"][()]])
      
        node_a_ms = node_mask[np.where(mask_ms==1)]
        node_a_mt2 = node_mask[np.where(mask_mt2==1)]
        node_a_art = node_mask[np.where((mask_mt2==1)|(mask_ms==1))]
        node_a_nart = node_mask[np.where((mask_mt2==0)&(mask_ms==0))]

    else:
        # MAIN BRANCHES

        # open the file
        if sim_dict['hist_split']==0:
            print("Main branch: opening file ",plot_dir+'mainbranch_'+sim_dict['sim']+'.hdf5')
            f = h5py.File(plot_dir+'mainbranch_'+sim_dict['sim']+'.hdf5','r')
            mass_b = f["/data/mass"][()]
            snap_b = f["/data/snapshot"][()]
            host_b = f["/data/host_flag"][()]
            if sim=="PMILL31":
                npartmax_raw = f["/data/max_npart_branch"][()]
                maxmass_raw = npartmax_raw*sim_dict['partmass']
            else:
                maxmass_raw = f["/data/max_mass_branch"][()]
                npartmax_raw = np.around(maxmass_raw/sim_dict['partmass'])
            finalID_raw = f["/data/nodeIndex_finalsnap"][()]
            finalhost_raw = f["/data/host_flag_finalsnap"][()]
            node_raw = f["/data/nodeIndex"][()]

        else:
            mass_b = np.array([],dtype='float32')
            snap_b = np.array([],dtype='int32')
            host_b = np.array([],dtype='int32')
            maxmass_raw = np.array([],dtype='float32')
            npartmax_raw = np.array([],dtype='int32')
            finalID_raw = np.array([],dtype='int64')
            finalhost_raw = np.array([],dtype='int64')
            node_raw = np.array([],dtype='int64')
            mbps_raw = np.array([],dtype='int32')

            if sim == 'PMILL31':
                ran = 31
            elif sim == 'FLAMINGO':
                ran = 64
            else:
                ran = sim_dict['dhalo_subvol']
            for i in range(ran):
                print("Main branch: opening file ",plot_dir+'mainbranch_'+sim_dict['sim']+'.'+str(i)+'.hdf5')
                f = h5py.File(plot_dir+'mainbranch_'+sim_dict['sim']+'.'+str(i)+'.hdf5','r')
                mass_b = np.concatenate([mass_b,f["/data/mass"][()]])
                snap_b = np.concatenate([snap_b,f["/data/snapshot"][()]])
                host_b = np.concatenate([host_b,f["/data/host_flag"][()]])
                if sim!="PMILL31":
                    maxmass_raw = np.concatenate([maxmass_raw,f["/data/max_mass_branch"][()]])
                    npartmax_raw = np.concatenate([npartmax_raw,np.around(maxmass_raw/sim_dict['partmass'])])
                else:
                    npartmax_raw = np.concatenate([npartmax_raw,f["/data/max_npart_branch"][()]])
                    maxmass_raw = np.concatenate([maxmass_raw,npartmax_raw*sim_dict['partmass']])
                finalID_raw = np.concatenate([finalID_raw,f["/data/nodeIndex_finalsnap"][()]])
                finalhost_raw = np.concatenate([finalhost_raw,f["/data/host_flag_finalsnap"][()]])
                node_raw = np.concatenate([node_raw,f["/data/nodeIndex"][()]])
                if allsubv==False:
                    break
        f.close()

        if rem_sat==True:
            # only account for main branches that are main halos at final snap
            ind_main = np.where(finalhost_raw==1) # select indices final snap which are main halos
            # arrays only with centrals
            mass = mass_b[ind_main]
            snap = snap_b[ind_main]
            finalID_raw = finalID_raw[ind_main]
            node = node_raw[ind_main]
        # differences in snaps
        # arrays are sorted by finalID (lower to higher) and by snap secondly (lower to higher) 
        dmass_raw = mass[1:]-mass[0:-1] # dmass
        massi_raw = mass[0:-1]
        massf_raw = mass[1:]
        snapi_raw = snap[0:-1]
        snapf_raw = snap[1:]
        dfinalID = finalID_raw[0:-1]-finalID_raw[1:]
        finalIDi_raw = finalID_raw[0:-1]
        nodei_raw = node[0:-1]
        nodef_raw = node[1:]

        ind = np.where(dfinalID==0)
        dmass = dmass_raw[ind]
        massi = massi_raw[ind]
        massf = massf_raw[ind]
        snapi = snapi_raw[ind]
        snapf = snapf_raw[ind]
        finalID = finalIDi_raw[ind]
        nodei = node_raw[0:-1]
        nodef = node_raw[1:]
        if np.shape(finalID_raw)[0]!=(np.shape(np.unique(finalID_raw))[0]+np.shape(dmass)[0]):
            raise Exception("CHECK has not passed") 

        # branches with issues:
        snap_list,z_list = np.loadtxt(sim_dict['zfile'],usecols=(0,1),unpack=True)
        ind_zi = np.searchsorted(snap_list,snapi)
        ind_zf = np.searchsorted(snap_list,snapf)
        redshift_i = z_list[ind_zi]
        redshift_f = z_list[ind_zf]
        # numerical artifact
        ind_ms = np.where(((massf>(np.exp(-alpha_up*(redshift_f-redshift_i))+offset_up)*massi))|(massf<(np.exp(-alpha_down*(redshift_f-redshift_i))+offset_down)*massi))
        # total fraction: unique IDs with issues/unique IDs 
        # numerical artifact
        finalID_ms = finalID[ind_ms] # affected ones
        uniqueID_ms, ind_uniqueID_ms = np.unique(finalID_ms,return_index=True) # only unique final IDs
        
        uniqueID,ind_uniqueID_raw,counts = np.unique(finalID_raw,return_index=True,return_counts=True)
        nodeend = node[ind_uniqueID_raw]
        massend = mass[ind_uniqueID_raw]
        npartend = np.around(massend/sim_dict['partmass']) # particle number halos born
        snapend = snap[ind_uniqueID_raw]
        finalIDend = finalID_raw[ind_uniqueID_raw]
        # branches with issues:
        # numerical artifact
        z_file_list,file_3sigma_list = np.loadtxt(sim_dict['3sigma_file'],usecols=(0,1),unpack=True)
        indsigma = np.where(file_3sigma_list<sim_dict['min_part']*10)
        file_3sigma_list[indsigma] = sim_dict['min_part']*10
        # group into redshift
        snap_list,z_list = np.loadtxt(sim_dict['zfile'],usecols=(0,1),unpack=True)
        ind_zi = np.searchsorted(snap_list,snapend)
        redshiftend = z_list[ind_zi]
        ind_dig = np.digitize(redshiftend,z_file_list)
        file_3sigma_end = file_3sigma_list[ind_dig]
        ind_mt2 = np.where(npartend>file_3sigma_end)
        finalID_mt2 = finalIDend[ind_mt2]

        # combine
        finalID_art = np.concatenate([finalID_ms,finalID_mt2])
        uniqueID_art = np.unique(finalID_art)

        # incorrect branches
        ind_ms = np.in1d(finalID_raw,uniqueID_ms)
        node_ms = node[ind_ms]
        snap_ms = snap[ind_ms]
        finalID_raw_ms = finalID_raw[ind_ms]
        ind_mt2 = np.in1d(finalID_raw,finalID_mt2)
        node_mt2 = node[ind_mt2]
        snap_mt2 = snap[ind_mt2]
        finalID_raw_mt2 = finalID_raw[ind_mt2]
        ind_art = np.in1d(finalID_raw,uniqueID_art)
        node_art = node[ind_art]
        snap_art = snap[ind_art]
        finalID_raw_art = finalID_raw[ind_art]
        # correct branches
        node_nart = node[~ind_art]
        snap_nart = snap[~ind_art]
        finalID_raw_nart = finalID_raw[~ind_art]


    # SHARK

    # open shark files
    fields = {'galaxies':('type','id_subhalo_tree','id_halo_tree','sfr_disk','sfr_burst','sfr_burst_diskins','sfr_burst_mergers',
                          'mvir_subhalo','mvir_hosthalo',"mstars_disk","mstars_bulge","m_bh","mstellar_halo","bh_accretion_rate_hh","bh_accretion_rate_sb","matom_disk","matom_bulge","id_galaxy")}
    fields_csfrd = {'galaxies':('type','id_subhalo_tree','id_halo_tree','sfr_disk','sfr_burst','sfr_burst_diskins','sfr_burst_mergers',"id_galaxy")}

    sfh_fields = {'bulges_diskins': ('star_formation_rate_histories'),
                  'bulges_mergers': ('star_formation_rate_histories'),
                  'disks': ('star_formation_rate_histories'),
                  'galaxies': ('id_galaxy')}

    z_out = np.array([0,0.5,1,2,3,5,7,10])
    #z_out = np.array([0])
    snap_list,z_list = np.loadtxt(sim_dict['zfile'],usecols=(0,1),unpack=True)
    isnap_a = np.array([],dtype='int32')
    
    BH_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BH_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BH_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BH_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BH_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SFR_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SFR_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SFR_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SFR_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SFR_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BHSFR_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BHSFR_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BHSFR_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BHSFR_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    BHSFR_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))
    massgal_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    massgal_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    massgal_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    massgal_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    massgal_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SMH1M_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SMH1M_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SMH1M_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SMH1M_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    SMH1M_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))
    HMF_a = np.zeros(shape = (len(z_out), len(xmf)))
    HMF_ms = np.zeros(shape = (len(z_out), len(xmf)))
    HMF_mt2 = np.zeros(shape = (len(z_out), len(xmf)))
    HMF_art = np.zeros(shape = (len(z_out), len(xmf)))
    HMF_nart = np.zeros(shape = (len(z_out), len(xmf)))
    SMF_a = np.zeros(shape = (len(z_out), len(xmf)))
    SMF_ms = np.zeros(shape = (len(z_out), len(xmf)))
    SMF_mt2 = np.zeros(shape = (len(z_out), len(xmf)))
    SMF_art = np.zeros(shape = (len(z_out), len(xmf)))
    SMF_nart = np.zeros(shape = (len(z_out), len(xmf)))
    CSFRD_a = np.zeros(shape = (len(z_list)))
    CSFRD_ms = np.zeros(shape = (len(z_list)))
    CSFRD_mt2 = np.zeros(shape = (len(z_list)))
    CSFRD_art = np.zeros(shape = (len(z_list)))
    CSFRD_nart = np.zeros(shape = (len(z_list)))
    mshH_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    mshH_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    mshH_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    mshH_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    mshH_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))
    eddBH_a = np.zeros(shape = (len(z_out), 3, len(xmf)))
    eddBH_ms = np.zeros(shape = (len(z_out), 3, len(xmf)))
    eddBH_mt2 = np.zeros(shape = (len(z_out), 3, len(xmf)))
    eddBH_art = np.zeros(shape = (len(z_out), 3, len(xmf)))
    eddBH_nart = np.zeros(shape = (len(z_out), 3, len(xmf)))


    for index in range(len(z_out)):

        isnap = np.argmin(abs(z_list-z_out[index]))
        isnap_a = np.append(isnap_a,isnap)
        if sim == 'PMILL31':
            hdf5_data = read_shark_data(sim_dict['shark_dir']+shark_model,int(snap_list[isnap]),fields,np.arange(31))
            if z_out[index]==0.:
                sfh, delta_t, LBT = read_shark_sfh(sim_dict['shark_dir']+shark_model,int(snap_list[isnap]),sfh_fields,np.arange(31))
                (bulge_diskins_hist, bulge_mergers_hist, disk_hist, id_sfh, uniqueid_sfh) = sfh
                sfhist = bulge_diskins_hist + bulge_mergers_hist + disk_hist
        elif sim == 'FLAMINGO':
            hdf5_data = read_shark_data(sim_dict['shark_dir']+shark_model,int(snap_list[isnap]),fields,np.arange(64))
            if z_out[index]==0.:
                sfh, delta_t, LBT = read_shark_sfh(sim_dict['shark_dir']+shark_model,int(snap_list[isnap]),sfh_fields,np.arange(64))
                (bulge_diskins_hist, bulge_mergers_hist, disk_hist, id_sfh, uniqueid_sfh) = sfh
                sfhist = bulge_diskins_hist + bulge_mergers_hist + disk_hist
        else:
            hdf5_data = read_shark_data(sim_dict['shark_dir']+shark_model,int(snap_list[isnap]),fields,np.arange(sim_dict['dhalo_subvol']))
            if z_out[index]==0.:
                sfh, delta_t, LBT = read_shark_sfh(sim_dict['shark_dir']+shark_model,int(snap_list[isnap]),sfh_fields,np.arange(sim_dict['dhalo_subvol']))
                (bulge_diskins_hist, bulge_mergers_hist, disk_hist, id_sfh, uniqueid_sfh) = sfh
                sfhist = bulge_diskins_hist + bulge_mergers_hist + disk_hist
        (h0,volh,typeg_a,sharkid_a,sharkhid_a,sfrd_a,sfrb_a,sfrbd_a,sfrbm_a,
         msubh_a,mh_a,mstars_disk_a,mstars_bulge_a,m_bh_a,msh_a,macc_hh_a,macc_sb_a,mh1_disk_a,mh1_bulge_a,id_a,uniqueid_a) = hdf5_data

        sfrd_a = sfrd_a/GyrToYr
        sfrb_a = sfrb_a/GyrToYr
        sfrbd_a = sfrbd_a/GyrToYr
        sfrbm_a = sfrbm_a/GyrToYr
        sfr_a = sfrd_a + sfrb_a
        mstars_a = (mstars_disk_a + mstars_bulge_a)
        macc_hh_a = macc_hh_a/GyrToYr
        macc_sb_a = macc_sb_a/GyrToYr
        macc_bh_a = macc_hh_a + macc_sb_a
        mh1_a = (mh1_disk_a + mh1_bulge_a)


        if all_branches == True:
            
            # ALL BRANCHES
            if include_indv == True:
                x1_a, y1_a, x2_a, y2_a, x3_a, y3_a, x4_a, y4_a, x7_a, y7_a = fill_arrays(h0, typeg_a,
                                                                                     m_bh_a, mstars_a, sfr_a, mh_a,
                                                                                     mh1_a, BH_a, SFR_a, BHSFR_a,
                                                                                         massgal_a, SMH1M_a, index, only_centrals,
                                                                                         only_centrals_hmf)
                x5_a, y5_a, x6_a, y6_a = fill_arrays_sh(h0, typeg_a, msh_a, mstars_a, mh_a, m_bh_a, macc_bh_a, mshH_a, eddBH_a, index, only_centrals)
            else:
                fill_arrays(h0, typeg_a, m_bh_a, mstars_a, sfr_a, mh_a, mh1_a, BH_a, SFR_a, BHSFR_a, massgal_a, SMH1M_a, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_a, msh_a, mstars_a, mh_a, m_bh_a, macc_bh_a, mshH_a, eddBH_a, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_a, mstars_a, mh_a, HMF_a, SMF_a, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_a, sfr_a, CSFRD_a, isnap, only_centrals)
            if z_out[index]==0:
                mbins_sfh = (9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5, 11.75, 12.5)
                SFH_med_a = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_ms = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_mt2 = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_art = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_nart = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))

                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_a)
                sfhist_a = sfhist[ind,:]
                ind = np.in1d(uniqueid_a,uniqueid_sfh)
                mstars_sfh = mstars_a[ind]
                typeg_sfh = typeg_a[ind]
                fill_arrays_sfh(h0, typeg_sfh, mstars_sfh, sfhist_a, LBT, SFH_med_a, only_centrals)

            # select the branches that are related to numerical artefacts
            # MS
            ind_sharkid = np.in1d(sharkid_a,node_a_ms)
            typeg_ms = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_ms))
            sharkid_ms = sharkid_a[ind_sharkid]
            sharkhid_ms = sharkhid_a[ind_sharkid]
            sfrd_ms = sfrd_a[ind_sharkid]
            sfrb_ms = sfrb_a[ind_sharkid]
            sfrbd_ms = sfrbd_a[ind_sharkid]
            sfrbm_ms = sfrbm_a[ind_sharkid]
            sfr_ms = sfrd_ms + sfrb_ms
            msubh_ms = msubh_a[ind_sharkid]
            mh_ms = mh_a[ind_sharkid]
            mstars_disk_ms = mstars_disk_a[ind_sharkid]
            mstars_bulge_ms = mstars_bulge_a[ind_sharkid]
            m_bh_ms = m_bh_a[ind_sharkid]
            mh1_ms = mh1_a[ind_sharkid]
            msh_ms = msh_a[ind_sharkid]
            macc_bh_ms = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_ms = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_a_ms,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_ms = sfrd_ms + sfrb_ms
            mstars_ms = (mstars_disk_ms + mstars_bulge_ms)
            if include_indv == True:
                x1_ms, y1_ms, x2_ms, y2_ms, x3_ms, y3_ms, x4_ms, y4_ms, x7_ms, y7_ms = fill_arrays(h0, typeg_ms,
                                                                                               m_bh_ms, mstars_ms, sfr_ms, mh_ms,
                                                                                               mh1_ms, BH_ms, SFR_ms, BHSFR_ms,
                                                                                                   massgal_ms, SMH1M_ms, index, only_centrals,
                                                                                                   only_centrals_hmf)
                x5_ms, y5_ms, x6_ms, y6_ms = fill_arrays_sh(h0, typeg_ms, msh_ms, mstars_ms, mh_ms, m_bh_ms, macc_bh_ms, mshH_ms, eddBH_ms, index, only_centrals)
            else:
                fill_arrays(h0, typeg_ms, m_bh_ms, mstars_ms, sfr_ms, mh_ms, mh1_ms, BH_ms, SFR_ms, BHSFR_ms, massgal_ms, SMH1M_ms, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_ms, msh_ms, mstars_ms, mh_ms, m_bh_ms, macc_bh_ms, mshH_ms, eddBH_ms, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_ms, mstars_ms, mh_ms, HMF_ms, SMF_ms, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_ms, sfr_ms, CSFRD_ms, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_ms)
                sfhist_ms = sfhist[ind,:]
                ind = np.in1d(uniqueid_ms,uniqueid_sfh)
                mstars_sfh_ms = mstars_ms[ind]
                typeg_sfh_ms = typeg_ms[ind]
                fill_arrays_sfh(h0, typeg_sfh_ms, mstars_sfh_ms, sfhist_ms, LBT, SFH_med_ms, only_centrals)

            # MT
            ind_sharkid = np.in1d(sharkid_a,node_a_mt2)
            typeg_mt2 = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_mt2))
            sharkid_mt2 = sharkid_a[ind_sharkid]
            sharkhid_mt2 = sharkhid_a[ind_sharkid]
            sfrd_mt2 = sfrd_a[ind_sharkid]
            sfrb_mt2 = sfrb_a[ind_sharkid]
            sfrbd_mt2 = sfrbd_a[ind_sharkid]
            sfrbm_mt2 = sfrbm_a[ind_sharkid]
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            msubh_mt2 = msubh_a[ind_sharkid]
            mh_mt2 = mh_a[ind_sharkid]
            mstars_disk_mt2 = mstars_disk_a[ind_sharkid]
            mstars_bulge_mt2 = mstars_bulge_a[ind_sharkid]
            m_bh_mt2 = m_bh_a[ind_sharkid]
            mh1_mt2 = mh1_a[ind_sharkid]
            msh_mt2 = msh_a[ind_sharkid]
            macc_bh_mt2 = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_mt2 = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_a_mt2,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            mstars_mt2 = (mstars_disk_mt2 + mstars_bulge_mt2)
            if include_indv == True:
                x1_mt2, y1_mt2, x2_mt2, y2_mt2, x3_mt2, y3_mt2, x4_mt2, y4_mt2, x7_mt2, y7_mt2 = fill_arrays(h0, typeg_mt2,
                                                                                                         m_bh_mt2, mstars_mt2, sfr_mt2, mh_mt2,
                                                                                                         mh1_mt2, BH_mt2, SFR_mt2, BHSFR_mt2,
                                                                                                             massgal_mt2, SMH1M_mt2, index, only_centrals,
                                                                                                             only_centrals_hmf)
                x5_mt2, y5_mt2, x6_mt2, y6_mt2 = fill_arrays_sh(h0, typeg_mt2, msh_mt2, mstars_mt2, mh_mt2, m_bh_mt2, macc_bh_mt2, mshH_mt2, eddBH_mt2, index, only_centrals)
            else:
                fill_arrays(h0, typeg_mt2, m_bh_mt2, mstars_mt2, sfr_mt2, mh_mt2, mh1_mt2, BH_mt2, SFR_mt2, BHSFR_mt2, massgal_mt2, SMH1M_mt2, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_mt2, msh_mt2, mstars_mt2, mh_mt2, m_bh_mt2, macc_bh_mt2, mshH_mt2, eddBH_mt2, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_mt2, mstars_mt2, mh_mt2, HMF_mt2, SMF_mt2, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_mt2, sfr_mt2, CSFRD_mt2, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_mt2)
                sfhist_mt2 = sfhist[ind,:]
                ind = np.in1d(uniqueid_mt2,uniqueid_sfh)
                mstars_sfh_mt2 = mstars_mt2[ind]
                typeg_sfh_mt2 = typeg_mt2[ind]
                fill_arrays_sfh(h0, typeg_sfh_mt2, mstars_sfh_mt2, sfhist_mt2, LBT, SFH_med_mt2, only_centrals)

            # MS + MT
            ind_sharkid = np.in1d(sharkid_a,node_a_art)
            typeg_art = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_art))
            sharkid_art = sharkid_a[ind_sharkid]
            sharkhid_art = sharkhid_a[ind_sharkid]
            sfrd_art = sfrd_a[ind_sharkid]
            sfrb_art = sfrb_a[ind_sharkid]
            sfrbd_art = sfrbd_a[ind_sharkid]
            sfrbm_art = sfrbm_a[ind_sharkid]
            sfr_art = sfrd_art + sfrb_art
            msubh_art = msubh_a[ind_sharkid]
            mh_art = mh_a[ind_sharkid]
            mstars_disk_art = mstars_disk_a[ind_sharkid]
            mstars_bulge_art = mstars_bulge_a[ind_sharkid]
            m_bh_art = m_bh_a[ind_sharkid]
            mh1_art = mh1_a[ind_sharkid]
            msh_art = msh_a[ind_sharkid]
            macc_bh_art = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_art = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_a_art,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_art = sfrd_art + sfrb_art
            mstars_art = (mstars_disk_art + mstars_bulge_art)
            if include_indv == True:
                x1_art, y1_art, x2_art, y2_art, x3_art, y3_art, x4_art, y4_art, x7_art, y7_art = fill_arrays(h0, typeg_art,
                                                                                                         m_bh_art,mstars_art, sfr_art, mh_art,
                                                                                                         mh1_art,BH_art, SFR_art, BHSFR_art,
                                                                                                             massgal_art, SMH1M_art, index, only_centrals,
                                                                                                             only_centrals_hmf)
                x5_art, y5_art, x6_art, y6_art = fill_arrays_sh(h0, typeg_art, msh_art, mstars_art, mh_art, m_bh_art, macc_bh_art, mshH_art, eddBH_art, index, only_centrals)
            else:
                fill_arrays(h0, typeg_art, m_bh_art,mstars_art, sfr_art, mh_art, mh1_art,BH_art, SFR_art, BHSFR_art, massgal_art, SMH1M_art, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_art, msh_art, mstars_art, mh_art, m_bh_art, macc_bh_art, mshH_art, eddBH_art, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_art, mstars_art, mh_art, HMF_art, SMF_art, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_art, sfr_art, CSFRD_art, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_art)
                sfhist_art = sfhist[ind,:]
                ind = np.in1d(uniqueid_art,uniqueid_sfh)
                mstars_sfh_art = mstars_art[ind]
                typeg_sfh_art = typeg_art[ind]
                fill_arrays_sfh(h0, typeg_sfh_art, mstars_sfh_art, sfhist_art, LBT, SFH_med_art, only_centrals)

            # NO ARTEFACTS
            ind_sharkid = np.in1d(sharkid_a,node_a_nart)
            typeg_nart = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_nart))
            sharkid_nart = sharkid_a[ind_sharkid]
            sharkhid_nart = sharkhid_a[ind_sharkid]
            sfrd_nart = sfrd_a[ind_sharkid]
            sfrb_nart = sfrb_a[ind_sharkid]
            sfrbd_nart = sfrbd_a[ind_sharkid]
            sfrbm_nart = sfrbm_a[ind_sharkid]
            sfr_nart = sfrd_nart + sfrb_nart
            msubh_nart = msubh_a[ind_sharkid]
            mh_nart = mh_a[ind_sharkid]
            mstars_disk_nart = mstars_disk_a[ind_sharkid]
            mstars_bulge_nart = mstars_bulge_a[ind_sharkid]
            m_bh_nart = m_bh_a[ind_sharkid]
            msh_nart = msh_a[ind_sharkid]
            mh1_nart = mh1_a[ind_sharkid]
            macc_bh_nart = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_nart = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_a_nart,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_nart = sfrd_nart + sfrb_nart
            mstars_nart = (mstars_disk_nart + mstars_bulge_nart)
            if include_indv == True:
                x1_nart, y1_nart, x2_nart, y2_nart, x3_nart, y3_nart, x4_nart, y4_nart, x7_nart, y7_nart = fill_arrays(h0, typeg_nart,
                                                                                                                   m_bh_nart, mstars_nart, sfr_nart, mh_nart,
                                                                                                                   mh1_nart, BH_nart, SFR_nart, BHSFR_nart,
                                                                                                                       massgal_nart, SMH1M_nart, index, only_centrals,
                                                                                                                       only_centrals_hmf)
                x5_nart, y5_nart, x6_nart, y6_nart = fill_arrays_sh(h0, typeg_nart, msh_nart, mstars_nart, mh_nart, m_bh_nart, macc_bh_nart, mshH_nart, eddBH_nart, index, only_centrals)
            else:
                fill_arrays(h0, typeg_nart, m_bh_nart, mstars_nart, sfr_nart, mh_nart, mh1_nart, BH_nart, SFR_nart, BHSFR_nart, massgal_nart, SMH1M_nart, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_nart, msh_nart, mstars_nart, mh_nart, m_bh_nart, macc_bh_nart, mshH_nart, eddBH_nart, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_nart, mstars_nart, mh_nart, HMF_nart, SMF_nart, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_nart, sfr_nart, CSFRD_nart, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_nart)
                sfhist_nart = sfhist[ind,:]
                ind = np.in1d(uniqueid_nart,uniqueid_sfh)
                mstars_sfh_nart = mstars_nart[ind]
                typeg_sfh_nart = typeg_nart[ind]
                fill_arrays_sfh(h0, typeg_sfh_nart, mstars_sfh_nart, sfhist_nart, LBT, SFH_med_nart, only_centrals)


        else:

            # MAIN BRANCHES        
            print('All subhalos in simulation:',np.shape(node))
            print('All galaxies at redshift '+str(z_out[index])+' in simulation:',np.shape(sharkid_a))
            ind_sharkid = np.in1d(sharkid_a,node)
            typeg_mb = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with subhalos in simulation:',np.shape(typeg_mb))
            sharkid_mb = sharkid_a[ind_sharkid]
            sharkhid_mb = sharkhid_a[ind_sharkid]
            sfrd_mb = sfrd_a[ind_sharkid]
            sfrb_mb = sfrb_a[ind_sharkid]
            sfrbd_mb = sfrbd_a[ind_sharkid]
            sfrbm_mb = sfrbm_a[ind_sharkid]
            sfr_mb = sfrd_mb + sfrb_mb
            msubh_mb = msubh_a[ind_sharkid]
            mh_mb = mh_a[ind_sharkid]
            mstars_disk_mb = mstars_disk_a[ind_sharkid]
            mstars_bulge_mb = mstars_bulge_a[ind_sharkid]
            m_bh_mb = m_bh_a[ind_sharkid]
            msh_mb = msh_a[ind_sharkid]
            mh1_mb = mh1_a[ind_sharkid]
            macc_bh_mb = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_mb = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node,sharkid_a)
            print('All subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_mb = sfrd_mb + sfrb_mb
            mstars_mb = (mstars_disk_mb + mstars_bulge_mb)
            if include_indv == True:
                x1_a, y1_a, x2_a, y2_a, x3_a, y3_a, x4_a, y4_a, x7_a, y7_a = fill_arrays(h0, typeg_mb,
                                                                                     m_bh_mb, mstars_mb, sfr_mb, mh_mb,
                                                                                     mh1_mb, BH_a, SFR_a, BHSFR_a,
                                                                                         massgal_a, SMH1M_a, index, only_centrals,
                                                                                         only_centrals_hmf)
                x5_a, y5_a, x6_a, y6_a = fill_arrays_sh(h0, typeg_mb, msh_mb, mstars_mb, mh_mb, m_bh_mb, macc_bh_mb, mshH_a, eddBH_a, index, only_centrals)
            else:
                fill_arrays(h0, typeg_mb, m_bh_mb, mstars_mb, sfr_mb, mh_mb, mh1_mb, BH_a, SFR_a, BHSFR_a, massgal_a, SMH1M_a, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_mb, msh_mb, mstars_mb, mh_mb, m_bh_mb, macc_bh_mb, mshH_a, eddBH_a, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_mb, mstars_mb, mh_mb, HMF_a, SMF_a, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_mb, sfr_mb, CSFRD_a, isnap, only_centrals)
            if z_out[index]==0:
                mbins_sfh = (9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,11.25,11.5, 11.75, 12.5)
                SFH_med_a = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_ms = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_mt2 = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_art = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))
                SFH_med_nart = np.zeros(shape = (3,len(LBT),len(mbins_sfh)-1))

                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_mb)
                sfhist_mb = sfhist[ind,:]
                ind = np.in1d(uniqueid_mb,uniqueid_sfh)
                mstars_sfh_mb = mstars_mb[ind]
                typeg_sfh_mb = typeg_mb[ind]
                fill_arrays_sfh(h0, typeg_sfh_mb, mstars_sfh_mb, sfhist_mb, LBT, SFH_med_a, only_centrals)

            # MS MB
            ind_sharkid = np.in1d(sharkid_a,node_ms)
            typeg_ms = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_ms))
            sharkid_ms = sharkid_a[ind_sharkid]
            sharkhid_ms = sharkhid_a[ind_sharkid]
            sfrd_ms = sfrd_a[ind_sharkid]
            sfrb_ms = sfrb_a[ind_sharkid]
            sfrbd_ms = sfrbd_a[ind_sharkid]
            sfrbm_ms = sfrbm_a[ind_sharkid]
            sfr_ms = sfrd_ms + sfrb_ms
            msubh_ms = msubh_a[ind_sharkid]
            mh_ms = mh_a[ind_sharkid]
            mstars_disk_ms = mstars_disk_a[ind_sharkid]
            mstars_bulge_ms = mstars_bulge_a[ind_sharkid]
            m_bh_ms = m_bh_a[ind_sharkid]
            msh_ms = msh_a[ind_sharkid]
            mh1_ms = mh1_a[ind_sharkid]
            macc_bh_ms = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_ms = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_ms,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_ms = sfrd_ms + sfrb_ms
            mstars_ms = (mstars_disk_ms + mstars_bulge_ms)
            if include_indv == True:
                x1_ms, y1_ms, x2_ms, y2_ms, x3_ms, y3_ms, x4_ms, y4_ms, x7_ms, y7_ms = fill_arrays(h0, typeg_ms,
                                                                                               m_bh_ms, mstars_ms, sfr_ms, mh_ms,
                                                                                               mh1_ms, BH_ms, SFR_ms, BHSFR_ms,
                                                                                                   massgal_ms, SMH1M_ms, index, only_centrals,
                                                                                                   only_centrals_hmf)
                x5_ms, y5_ms, x6_ms, y6_ms = fill_arrays_sh(h0, typeg_ms, msh_ms, mstars_ms, mh_ms, m_bh_ms, macc_bh_ms, mshH_ms, eddBH_ms, index, only_centrals)
            else:
                fill_arrays(h0, typeg_ms, m_bh_ms, mstars_ms, sfr_ms, mh_ms, mh1_ms, BH_ms, SFR_ms, BHSFR_ms, massgal_ms, SMH1M_ms, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_ms, msh_ms, mstars_ms, mh_ms, m_bh_ms, macc_bh_ms, mshH_ms, eddBH_ms, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_ms, mstars_ms, mh_ms, HMF_ms, SMF_ms, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_ms, sfr_ms, CSFRD_ms, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_ms)
                sfhist_ms = sfhist[ind,:]
                ind = np.in1d(uniqueid_ms,uniqueid_sfh)
                mstars_sfh_ms = mstars_ms[ind]
                typeg_sfh_ms = typeg_ms[ind]
                fill_arrays_sfh(h0, typeg_sfh_ms, mstars_sfh_ms, sfhist_ms, LBT, SFH_med_ms, only_centrals)

            # MT MB
            ind_sharkid = np.in1d(sharkid_a,node_mt2)
            typeg_mt2 = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_mt2))
            sharkid_mt2 = sharkid_a[ind_sharkid]
            sharkhid_mt2 = sharkhid_a[ind_sharkid]
            sfrd_mt2 = sfrd_a[ind_sharkid]
            sfrb_mt2 = sfrb_a[ind_sharkid]
            sfrbd_mt2 = sfrbd_a[ind_sharkid]
            sfrbm_mt2 = sfrbm_a[ind_sharkid]
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            msubh_mt2 = msubh_a[ind_sharkid]
            mh_mt2 = mh_a[ind_sharkid]
            mstars_disk_mt2 = mstars_disk_a[ind_sharkid]
            mstars_bulge_mt2 = mstars_bulge_a[ind_sharkid]
            m_bh_mt2 = m_bh_a[ind_sharkid]
            mh1_mt2 = mh1_a[ind_sharkid]
            msh_mt2 = msh_a[ind_sharkid]
            macc_bh_mt2 = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_mt2 = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_mt2,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            mstars_mt2 = (mstars_disk_mt2 + mstars_bulge_mt2)
            if include_indv == True:
                x1_mt2, y1_mt2, x2_mt2, y2_mt2, x3_mt2, y3_mt2, x4_mt2, y4_mt2, x7_mt2, y7_mt2 = fill_arrays(h0, typeg_mt2,
                                                                                                         m_bh_mt2, mstars_mt2, sfr_mt2, mh_mt2,
                                                                                                         mh1_mt2, BH_mt2, SFR_mt2, BHSFR_mt2,
                                                                                                             massgal_mt2, SMH1M_mt2, index, only_centrals,
                                                                                                             only_centrals_hmf)
                x5_mt2, y5_mt2, x6_mt2, y6_mt2 = fill_arrays_sh(h0, typeg_mt2, msh_mt2, mstars_mt2, mh_mt2, m_bh_mt2, macc_bh_mt2, mshH_mt2, eddBH_mt2, index, only_centrals)
            else:
                fill_arrays(h0, typeg_mt2, m_bh_mt2, mstars_mt2, sfr_mt2, mh_mt2, mh1_mt2, BH_mt2, SFR_mt2, BHSFR_mt2, massgal_mt2, SMH1M_mt2, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_mt2, msh_mt2, mstars_mt2, mh_mt2, m_bh_mt2, macc_bh_mt2, mshH_mt2, eddBH_mt2, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_mt2, mstars_mt2, mh_mt2, HMF_mt2, SMF_mt2, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_mt2, sfr_mt2, CSFRD_mt2, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_mt2)
                sfhist_mt2 = sfhist[ind,:]
                ind = np.in1d(uniqueid_mt2,uniqueid_sfh)
                mstars_sfh_mt2 = mstars_mt2[ind]
                typeg_sfh_mt2 = typeg_mt2[ind]
                fill_arrays_sfh(h0, typeg_sfh_mt2, mstars_sfh_mt2, sfhist_mt2, LBT, SFH_med_mt2, only_centrals)

            # MS + MT MB
            ind_sharkid = np.in1d(sharkid_a,node_art)
            typeg_art = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_art))
            sharkid_art = sharkid_a[ind_sharkid]
            sharkhid_art = sharkhid_a[ind_sharkid]
            sfrd_art = sfrd_a[ind_sharkid]
            sfrb_art = sfrb_a[ind_sharkid]
            sfrbd_art = sfrbd_a[ind_sharkid]
            sfrbm_art = sfrbm_a[ind_sharkid]
            sfr_art = sfrd_art + sfrb_art
            msubh_art = msubh_a[ind_sharkid]
            mh_art = mh_a[ind_sharkid]
            mstars_disk_art = mstars_disk_a[ind_sharkid]
            mstars_bulge_art = mstars_bulge_a[ind_sharkid]
            m_bh_art = m_bh_a[ind_sharkid] 
            mh1_art = mh1_a[ind_sharkid] 
            msh_art = msh_a[ind_sharkid] 
            macc_bh_art = macc_bh_a[ind_sharkid] 
            if z_out[index]==0.:
                uniqueid_art = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_art,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_art = sfrd_art + sfrb_art
            mstars_art = (mstars_disk_art + mstars_bulge_art)
            if include_indv == True:
                x1_art, y1_art, x2_art, y2_art, x3_art, y3_art, x4_art, y4_art, x7_art, y7_art = fill_arrays(h0, typeg_art,
                                                                                                         m_bh_art,mstars_art, sfr_art, mh_art,
                                                                                                         mh1_art,BH_art, SFR_art, BHSFR_art,
                                                                                                             massgal_art, SMH1M_art, index, only_centrals,
                                                                                                             only_centrals_hmf)
                x5_art, y5_art, x6_art, y6_art = fill_arrays_sh(h0, typeg_art, msh_art, mstars_art, mh_art, m_bh_art, macc_bh_art, mshH_art, eddBH_art, index, only_centrals)
            else:
                fill_arrays(h0, typeg_art, m_bh_art,mstars_art, sfr_art, mh_art, mh1_art,BH_art, SFR_art, BHSFR_art, massgal_art, SMH1M_art, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_art, msh_art, mstars_art, mh_art, m_bh_art, macc_bh_art, mshH_art, eddBH_art, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_art, mstars_art, mh_art, HMF_art, SMF_art, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_art, sfr_art, CSFRD_art, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_art)
                sfhist_art = sfhist[ind,:]
                ind = np.in1d(uniqueid_art,uniqueid_sfh)
                mstars_sfh_art = mstars_art[ind]
                typeg_sfh_art = typeg_art[ind]
                fill_arrays_sfh(h0, typeg_sfh_art, mstars_sfh_art, sfhist_art, LBT, SFH_med_art, only_centrals)

            # NO ARTEFACTS MB
            ind_sharkid = np.in1d(sharkid_a,node_nart)
            typeg_nart = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_nart))
            sharkid_nart = sharkid_a[ind_sharkid]
            sharkhid_nart = sharkhid_a[ind_sharkid]
            sfrd_nart = sfrd_a[ind_sharkid]
            sfrb_nart = sfrb_a[ind_sharkid]
            sfrbd_nart = sfrbd_a[ind_sharkid]
            sfrbm_nart = sfrbm_a[ind_sharkid]
            sfr_nart = sfrd_nart + sfrb_nart
            msubh_nart = msubh_a[ind_sharkid]
            mh_nart = mh_a[ind_sharkid]
            mstars_disk_nart = mstars_disk_a[ind_sharkid]
            mstars_bulge_nart = mstars_bulge_a[ind_sharkid]
            m_bh_nart = m_bh_a[ind_sharkid]
            mh1_nart = mh1_a[ind_sharkid]
            msh_nart = msh_a[ind_sharkid]
            macc_bh_nart = macc_bh_a[ind_sharkid]
            if z_out[index]==0.:
                uniqueid_nart = uniqueid_a[ind_sharkid]
            ind_node = np.in1d(node_nart,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_out[index])+':',np.shape(ind_node))
            sfr_nart = sfrd_nart + sfrb_nart
            mstars_nart = (mstars_disk_nart + mstars_bulge_nart)
            if include_indv == True:
                x1_nart, y1_nart, x2_nart, y2_nart, x3_nart, y3_nart, x4_nart, y4_nart, x7_nart, y7_nart = fill_arrays(h0, typeg_nart,
                                                                                                                   m_bh_nart,mstars_nart, sfr_nart, mh_nart,
                                                                                                                   mh1_nart, BH_nart, SFR_nart, BHSFR_nart,
                                                                                                                       massgal_nart, SMH1M_nart, index, only_centrals, only_centrals_hmf)
                x5_nart, y5_nart, x6_nart, y6_nart = fill_arrays_sh(h0, typeg_nart, msh_nart, mstars_nart, mh_nart, m_bh_nart, macc_bh_nart, mshH_nart, eddBH_nart, index, only_centrals)
            else:
                fill_arrays(h0, typeg_nart, m_bh_nart,mstars_nart, sfr_nart, mh_nart, mh1_nart, BH_nart, SFR_nart, BHSFR_nart, massgal_nart, SMH1M_nart, index, only_centrals, only_centrals_hmf)
                fill_arrays_sh(h0, typeg_nart, msh_nart, mstars_nart, mh_nart, m_bh_nart, macc_bh_nart, mshH_nart, eddBH_nart, index, only_centrals)
            fill_arrays_vol(h0, volh, typeg_nart, mstars_nart, mh_nart, HMF_nart, SMF_nart, index, only_centrals, only_centrals_hmf)
            fill_arrays_csfrd(h0, volh, typeg_nart, sfr_nart, CSFRD_nart, isnap, only_centrals)
            if z_out[index]==0:
                # select only main branches that are in both catalogues
                ind = np.in1d(uniqueid_sfh,uniqueid_nart)
                sfhist_nart = sfhist[ind,:]
                ind = np.in1d(uniqueid_nart,uniqueid_sfh)
                mstars_sfh_nart = mstars_nart[ind]
                typeg_sfh_nart = typeg_nart[ind]
                fill_arrays_sfh(h0, typeg_sfh_nart, mstars_sfh_nart, sfhist_nart, LBT, SFH_med_nart, only_centrals)

        if include_indv == True:
            plot_mstars_BH(plt, outdir, BH_a, BH_ms, BH_mt2 , BH_art, BH_nart, z_out[index], index,
                           only_centrals, all_branches,x1_ms,y1_ms,x1_mt2,y1_mt2,x1_nart,y1_nart)
            plot_mstars_SFR(plt, outdir, h0, SFR_a, SFR_art, SFR_nart, z_out[index], index,
                            only_centrals, all_branches, x2_ms,y2_ms,x2_mt2,y2_mt2,x2_nart,y2_nart)
            plot_BH_SSFR(plt, outdir, BHSFR_a, BHSFR_art, BHSFR_nart, z_out[index], index,
                         only_centrals, all_branches, x3_ms,y3_ms,x3_mt2,y3_mt2,x3_nart,y3_nart)
            plot_HM_SM(plt, outdir, massgal_a, massgal_art, massgal_nart, z_out[index], index,
                       only_centrals_hmf, all_branches, x4_ms,y4_ms,x4_mt2,y4_mt2,x4_nart,y4_nart)
            plot_SM_H1M(plt, outdir, SMH1M_a, SMH1M_art, SMH1M_nart, z_out[index], index,
                        only_centrals, all_branches, x7_ms,y7_ms,x7_mt2,y7_mt2,x7_nart,y7_nart)
            plot_HM_SHM(plt, outdir, mshH_a, mshH_art, mshH_nart, z_out[index], index,
                        only_centrals, all_branches, x5_ms,y5_ms,x5_mt2,y5_mt2,x5_nart,y5_nart)
            plot_BH_EDD(plt, outdir, eddBH_a, eddBH_art, eddBH_nart, z_out[index], index,
                    only_centrals, all_branches, x6_ms,y6_ms,x6_mt2,y6_mt2,x6_nart,y6_nart)
        else:
            plot_mstars_BH(plt, outdir, BH_a, BH_ms, BH_mt2 , BH_art, BH_nart, z_out[index], index,
                           only_centrals, all_branches)
            plot_mstars_SFR(plt, outdir, h0, SFR_a, SFR_art, SFR_nart, z_out[index], index,
                            only_centrals, all_branches)
            plot_BH_SSFR(plt, outdir, BHSFR_a, BHSFR_art, BHSFR_nart, z_out[index], index,
                         only_centrals, all_branches)
            plot_HM_SM(plt, outdir, massgal_a, massgal_art, massgal_nart, z_out[index], index,
                       only_centrals_hmf, all_branches)
            plot_SM_H1M(plt, outdir, SMH1M_a, SMH1M_art, SMH1M_nart, z_out[index], index,
                        only_centrals, all_branches)
            plot_HM_SHM(plt, outdir, mshH_a, mshH_art, mshH_nart, z_out[index], index,
                        only_centrals, all_branches)
            plot_BH_EDD(plt, outdir, eddBH_a, eddBH_art, eddBH_nart, z_out[index], index,
                    only_centrals, all_branches)
        
        plot_HMF(plt, outdir, h0, omega_m, HMF_a, HMF_art, HMF_nart, z_out[index], index,
         only_centrals_hmf, all_branches)
        plot_SMF(plt, outdir, h0, SMF_a, SMF_art, SMF_nart, z_out[index], index,
         only_centrals, all_branches)
        
        if z_out[index]==0:
            for j in range(0,len(mbins_sfh)-1):
                plot_individual_seds(plt, outdir, LBT, SFH_med_a, SFH_med_ms, SFH_med_mt2, SFH_med_art, SFH_med_nart, j, only_centrals, all_branches)

    ## NEW PART

    for index_z in range(len(z_list)):

        if index_z==0 or np.isin(index_z,isnap_a)==True:
            continue
        
        if sim == 'PMILL31':
            hdf5_data = read_shark_data(sim_dict['shark_dir']+shark_model,int(snap_list[index_z]),fields_csfrd,np.arange(31))
        elif sim == 'FLAMINGO':
            hdf5_data = read_shark_data(sim_dict['shark_dir']+shark_model,int(snap_list[index_z]),fields_csfrd,np.arange(64))
        else:
            hdf5_data = read_shark_data(sim_dict['shark_dir']+shark_model,int(snap_list[index_z]),fields_csfrd,np.arange(sim_dict['dhalo_subvol']))
        (h0,volh,typeg_a,sharkid_a,sharkhid_a,sfrd_a,sfrb_a,sfrbd_a,sfrbm_a,id_a,uniqueid_a) = hdf5_data

        sfrd_a = sfrd_a/GyrToYr
        sfrb_a = sfrb_a/GyrToYr
        sfrbd_a = sfrbd_a/GyrToYr
        sfrbm_a = sfrbm_a/GyrToYr
        sfr_a = sfrd_a + sfrb_a


        if all_branches == True:
            
            # ALL BRANCHES
            fill_arrays_csfrd(h0, volh, typeg_a, sfr_a, CSFRD_a, index_z, only_centrals)

            # select the branches that are related to numerical artefacts
            # MS
            ind_sharkid = np.in1d(sharkid_a,node_a_ms)
            typeg_ms = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_ms))
            sharkid_ms = sharkid_a[ind_sharkid]
            sharkhid_ms = sharkhid_a[ind_sharkid]
            sfrd_ms = sfrd_a[ind_sharkid]
            sfrb_ms = sfrb_a[ind_sharkid]
            sfrbd_ms = sfrbd_a[ind_sharkid]
            sfrbm_ms = sfrbm_a[ind_sharkid]
            sfr_ms = sfrd_ms + sfrb_ms
            ind_node = np.in1d(node_a_ms,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_ms = sfrd_ms + sfrb_ms
            fill_arrays_csfrd(h0, volh, typeg_ms, sfr_ms, CSFRD_ms, index_z, only_centrals)

            # MT
            ind_sharkid = np.in1d(sharkid_a,node_a_mt2)
            typeg_mt2 = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_mt2))
            sharkid_mt2 = sharkid_a[ind_sharkid]
            sharkhid_mt2 = sharkhid_a[ind_sharkid]
            sfrd_mt2 = sfrd_a[ind_sharkid]
            sfrb_mt2 = sfrb_a[ind_sharkid]
            sfrbd_mt2 = sfrbd_a[ind_sharkid]
            sfrbm_mt2 = sfrbm_a[ind_sharkid]
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            ind_node = np.in1d(node_a_mt2,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            fill_arrays_csfrd(h0, volh, typeg_mt2, sfr_mt2, CSFRD_mt2, index_z, only_centrals)

            # MS + MT
            ind_sharkid = np.in1d(sharkid_a,node_a_art)
            typeg_art = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_art))
            sharkid_art = sharkid_a[ind_sharkid]
            sharkhid_art = sharkhid_a[ind_sharkid]
            sfrd_art = sfrd_a[ind_sharkid]
            sfrb_art = sfrb_a[ind_sharkid]
            sfrbd_art = sfrbd_a[ind_sharkid]
            sfrbm_art = sfrbm_a[ind_sharkid]
            sfr_art = sfrd_art + sfrb_art
            ind_node = np.in1d(node_a_art,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_art = sfrd_art + sfrb_art
            fill_arrays_csfrd(h0, volh, typeg_art, sfr_art, CSFRD_art, index_z, only_centrals)

            # NO ARTEFACTS
            ind_sharkid = np.in1d(sharkid_a,node_a_nart)
            typeg_nart = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_nart))
            sharkid_nart = sharkid_a[ind_sharkid]
            sharkhid_nart = sharkhid_a[ind_sharkid]
            sfrd_nart = sfrd_a[ind_sharkid]
            sfrb_nart = sfrb_a[ind_sharkid]
            sfrbd_nart = sfrbd_a[ind_sharkid]
            sfrbm_nart = sfrbm_a[ind_sharkid]
            sfr_nart = sfrd_nart + sfrb_nart
            ind_node = np.in1d(node_a_nart,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_nart = sfrd_nart + sfrb_nart
            fill_arrays_csfrd(h0, volh, typeg_nart, sfr_nart, CSFRD_nart, index_z, only_centrals)


        else:

            # MAIN BRANCHES        
            print('All subhalos in simulation:',np.shape(node))
            print('All galaxies at redshift '+str(z_list[index_z])+' in simulation:',np.shape(sharkid_a))
            ind_sharkid = np.in1d(sharkid_a,node)
            typeg_mb = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with subhalos in simulation:',np.shape(typeg_mb))
            sharkid_mb = sharkid_a[ind_sharkid]
            sharkhid_mb = sharkhid_a[ind_sharkid]
            sfrd_mb = sfrd_a[ind_sharkid]
            sfrb_mb = sfrb_a[ind_sharkid]
            sfrbd_mb = sfrbd_a[ind_sharkid]
            sfrbm_mb = sfrbm_a[ind_sharkid]
            sfr_mb = sfrd_mb + sfrb_mb
            ind_node = np.in1d(node,sharkid_a)
            print('All subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_mb = sfrd_mb + sfrb_mb
            fill_arrays_csfrd(h0, volh, typeg_mb, sfr_mb, CSFRD_a, index_z, only_centrals)

            # MS MB
            ind_sharkid = np.in1d(sharkid_a,node_ms)
            typeg_ms = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_ms))
            sharkid_ms = sharkid_a[ind_sharkid]
            sharkhid_ms = sharkhid_a[ind_sharkid]
            sfrd_ms = sfrd_a[ind_sharkid]
            sfrb_ms = sfrb_a[ind_sharkid]
            sfrbd_ms = sfrbd_a[ind_sharkid]
            sfrbm_ms = sfrbm_a[ind_sharkid]
            sfr_ms = sfrd_ms + sfrb_ms
            ind_node = np.in1d(node_ms,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_ms = sfrd_ms + sfrb_ms
            fill_arrays_csfrd(h0, volh, typeg_ms, sfr_ms, CSFRD_ms, index_z, only_centrals)

            # MT MB
            ind_sharkid = np.in1d(sharkid_a,node_mt2)
            typeg_mt2 = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_mt2))
            sharkid_mt2 = sharkid_a[ind_sharkid]
            sharkhid_mt2 = sharkhid_a[ind_sharkid]
            sfrd_mt2 = sfrd_a[ind_sharkid]
            sfrb_mt2 = sfrb_a[ind_sharkid]
            sfrbd_mt2 = sfrbd_a[ind_sharkid]
            sfrbm_mt2 = sfrbm_a[ind_sharkid]
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            ind_node = np.in1d(node_mt2,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_mt2 = sfrd_mt2 + sfrb_mt2
            fill_arrays_csfrd(h0, volh, typeg_mt2, sfr_mt2, CSFRD_mt2, index_z, only_centrals)

            # MS + MT MB
            ind_sharkid = np.in1d(sharkid_a,node_art)
            typeg_art = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_out[index])+' with MS subhalos in simulation:',np.shape(typeg_art))
            sharkid_art = sharkid_a[ind_sharkid]
            sharkhid_art = sharkhid_a[ind_sharkid]
            sfrd_art = sfrd_a[ind_sharkid]
            sfrb_art = sfrb_a[ind_sharkid]
            sfrbd_art = sfrbd_a[ind_sharkid]
            sfrbm_art = sfrbm_a[ind_sharkid]
            sfr_art = sfrd_art + sfrb_art
            ind_node = np.in1d(node_art,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_art = sfrd_art + sfrb_art
            fill_arrays_csfrd(h0, volh, typeg_art, sfr_art, CSFRD_art, index_z, only_centrals)

            # NO ARTEFACTS MB
            ind_sharkid = np.in1d(sharkid_a,node_nart)
            typeg_nart = typeg_a[ind_sharkid]
            print('All galaxies at redshift '+str(z_list[index_z])+' with MS subhalos in simulation:',np.shape(typeg_nart))
            sharkid_nart = sharkid_a[ind_sharkid]
            sharkhid_nart = sharkhid_a[ind_sharkid]
            sfrd_nart = sfrd_a[ind_sharkid]
            sfrb_nart = sfrb_a[ind_sharkid]
            sfrbd_nart = sfrbd_a[ind_sharkid]
            sfrbm_nart = sfrbm_a[ind_sharkid]
            sfr_nart = sfrd_nart + sfrb_nart
            ind_node = np.in1d(node_nart,sharkid_a)
            print('All MS subhalos in simulation with galaxies at redshift '+str(z_list[index_z])+':',np.shape(ind_node))
            sfr_nart = sfrd_nart + sfrb_nart
            fill_arrays_csfrd(h0, volh, typeg_nart, sfr_nart, CSFRD_nart, index_z, only_centrals)
        
    plot_CSFRD(plt, outdir, h0, z_list, CSFRD_a, CSFRD_ms, CSFRD_mt2, CSFRD_art, CSFRD_nart, only_centrals, all_branches)
            

if __name__ == "__main__":
    main()


    

    
