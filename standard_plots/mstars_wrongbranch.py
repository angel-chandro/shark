#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""HMF plots"""
import collections
import logging
import os
import sys

import numpy as np
import h5py
import common
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mpl_style
from mymodule.main_branch_info import choose_sim

logger = logging.getLogger(__name__)

plt.style.use(mpl_style.style1)
#plt.rcParams['figure.figsize'] = (36, 27)
plt.rcParams['figure.figsize'] = (32, 24)
#np.set_printoptions(threshold=sys.maxsize)
#plt.rcParams['axes.labelsize'] = 2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['xtick.minor.pad'] = 2
plt.rcParams['ytick.minor.pad'] = 2
np.set_printoptions(threshold=sys.maxsize)

## more plotting options
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

linw1 = 3
linw2 = 2
linw3 = 15

size1 = 12
size2 = 40

##################################
mlow = 10
mupp = 15
dm = 0.2
mbins = np.arange(mlow,mupp,dm)
xmf = mbins + dm/2.0

# Constants
GyrToYr = 1e9
zsun = 0.0189
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s

# MS 
#ID = [78000002299080,78000046479704,78000061812879]
#snap = [[71],[49],[69]]

#ID = [78000006729505,78000027068764,78000023512096]
#snap = [[58],[38],[70]]

#ID = [78000029263059,78000048650684,78000003133049]
#snap = [[54],[71],[56]]

# MT
#ID = [78000040836418,78000058463701,78000064620801]
#snap = [[72],[62],[50]]

#ID = [78000031075336,78000061632859,78000053737695]
#snap = [[43],[31],[64]]

#ID = [78000018965657,78000033660252,78000049808992]
#snap = [[45],[37],[47]]

# FINAL MS
#ID = [78000061812879,78000027068764,78000029263059]
#snap = [[69],[38],[54]]
#snap = [[70],[39],[55]]

# FINAL MT
ID = [78000058463701,78000018965657,78000049808992]
snap = [[62],[43],[47]]


simu = "FLAMINGO"
sim_dict = choose_sim(simu)
snap_list = np.arange(1,sim_dict['snapshots'],1)

def read_sfh_subv(model_dir, snapshot, fields, subvolumes, include_h0_volh=True):
    """Read the galaxies.hdf5 file for the given model/snapshot/subvolume"""

    data = collections.OrderedDict()
    subv_a = np.array([],dtype='int32')
    for idx, subv in enumerate(subvolumes):

        fname = os.path.join(model_dir, str(snapshot), str(subv), 'star_formation_histories.hdf5')
        logger.info('Reading SFH data from %s', fname)
        with h5py.File(fname, 'r') as f:
            if idx == 0:
                delta_t = f['delta_t'][()]
                LBT     = f['lbt_mean'][()]

            j = 0 
            for gnames, dsname in fields.items():
                group = f[gnames]
                full_name = '%s/%s' % (gnames, dsname)
                l = data.get(full_name, None)
                if j==0:
                    gshape = np.shape(group[dsname][()])[0]
                    j += 1
                if l is None:
                    l = group[dsname][()]
                else:
                    l = np.concatenate([l, group[dsname][()]])
                data[full_name] = l
        subv_a = np.concatenate([subv_a,int(subv)*np.ones(gshape)])
    return list(data.values()), delta_t, LBT, subv_a

def plot_mah_snap(plt, outdir, obsdir, h0, mshalo, mhalo, mbh, mgas_disk, mhot, rgas_disk, bh_accretion_hh, bh_accretion_sb, matom, mmol, mbh_accreted, mean_stellar_age, mlost, mreheated, mgas_bulge, rgas_bulge, gal_props_z0, LBT, redshift, total_sfh_z0, sfh_b , sfh_c, sfh_d, mdisk, mbulge, matom_bulge, matom_disk, mmol_bulge, mmol_disk, vmax_sh, vvir_h, vvir_sh):

    xtit="Snapshot"
    ytit="$\\rm log_{10}(M_{subhalo}/(M_{\odot}/h))$"
    ytit2="$\\rm log_{10}(SFR/M_{\odot} yr^{-1})$"
    ytit3="$\\rm log_{10}(M_{stars}/(M_{\odot}/h))$"
    ytit4="$\\rm log_{10}(M_{hot}/(M_{\odot}/h))$"
    ytit5="$\\rm log_{10}(M_{gas}/(M_{\odot}/h))$"
    ytit6="$\\rm log_{10}(R_{gas,disk}/(Mpc/h))$"
    ytit7="$\\rm log_{10}(R_{gas,bulge}/(Mpc/h))$"
    ytit8="$\\rm log_{10}(BHAR/M_{\odot} yr^{-1})$"
    ytit9="$\\rm log_{10}(M_{BH}/(M_{\odot}/h))$"
    ytit10="$\\rm log_{10}(M_{H1}/(M_{\odot}/h))$"
    ytit11="$\\rm log_{10}(M_{H2}/(M_{\odot}/h))$"
    ytit12="$\\rm log_{10}(M_{gas,feedback}/(M_{\odot}/h))$"
    ytit13="$\\rm log_{10}(v/(km/s))$"

    xmin, xmax, ymin, ymax = 0, sim_dict['snapshots']-1, 9.8, 15
    ymin2, ymax2 = -7.8, 4.4
    ymin3, ymax3 = 1.6, 13.7
    ymin4, ymax4 = 7.6, 14.7
    ymin5, ymax5 = 3.8, 11.2
    ymin6, ymax6 = -4.6, -1.2
    ymin7, ymax7 = -4.6, -1.2
    ymin8, ymax8 = -10, 5
    ymin9, ymax9 = 3.6, 10.4
    ymin10, ymax10 = 2, 12
    ymin11, ymax11 = 1, 12
    ymin12, ymax12 = 0, 15
    ymin13, ymax13 = 1.6, 3.2

    xleg = xmin + 1 * (xmax-xmin)
    yleg = ymax - 0.07 * (ymax-ymin)

    fig = plt.figure(figsize=(size1,size2))
    gs1 = gridspec.GridSpec(13, 3)
    gs1.update(wspace=0, hspace=0)

    for i in range(39):
        
        ax = plt.subplot(gs1[i])

        if i==1 or i==2:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
        elif i==4 or i==5:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin2,ymax2)
        elif i==7 or i==8:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin3,ymax3)
        elif i==10 or i==11:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin4,ymax4)
        elif i==13 or i==14:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin5,ymax5)
        elif i==16 or i==17:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin6,ymax6)
        elif i==19 or i==20:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin7,ymax7)
        elif i==22 or i==23:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin8,ymax8)
        elif i==25 or i==26:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin9,ymax9)
        elif i==28 or i==29:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin10,ymax10)
        elif i==31 or i==32:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin11,ymax11)
        elif i==34 or i==35:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks(np.arange(0,len(LBT)+1,10))
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin12,ymax12)
        elif i==0:
            ax.set_xticklabels([])
            ax.set_ylabel(ytit,fontsize=linw3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
        elif i==3:
            ax.set_xticklabels([])
            ax.set_ylabel(ytit2,fontsize=linw3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin2,ymax2)
        elif i==6:
            ax.set_xticklabels([])
            ax.set_ylabel(ytit3,fontsize=linw3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin3,ymax3)
        elif i==9:
            ax.set_xticklabels([])
            ax.set_ylabel(ytit4,fontsize=linw3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin4,ymax4)
        elif i==12:
            ax.set_xticklabels([])
            ax.set_ylabel(ytit5,fontsize=linw3)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin5,ymax5)
        elif i==15:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit6,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin6,ymax6)
        elif i==18:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit7,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin7,ymax7)
        elif i==21:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit8,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin8,ymax8)
        elif i==24:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit9,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin9,ymax9)
        elif i==27:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit10,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin10,ymax10)
        elif i==30:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit11,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin11,ymax11)
        elif i==33:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit12,fontsize=linw3)            
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin12,ymax12)
        elif i==36:
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_ylabel(ytit13,fontsize=linw3)            
            ax.set_xticks(np.arange(0,len(LBT)+1,10))
            ax.set_xticklabels(np.arange(0,len(LBT)+1,10))
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin13,ymax13)
        elif i==37 or i==38:
            ax.set_yticklabels([])
            ax.set_xlabel(xtit,fontsize=linw3)
            ax.set_xticks(np.arange(0,len(LBT)+1,10))
            ax.set_xticklabels(np.arange(0,len(LBT)+1,10))
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin13,ymax13)

#        if i<3:
#            ind = np.where(mhalo[i,:]==0)
#            if np.shape(ind)[1]==1 and ind[0]!=0:
#                ind = ind[0]
#                mshalo[i,ind[0]] = mshalo[i,ind[0]-1]
#                mshalo[i,ind[0]] = mshalo[i,ind[0]-1]
#            elif np.shape(ind)[1]>1:
#                ind = ind[0]
#                h = ind[1:]-ind[0:-1]
#                ind2 = np.where(h!=1)
#                ind2 = ind2[0]+1
#                if np.shape(ind2)[0]>0:
#                    ind3 = np.where(ind==ind2)
#                    mshalo[i,ind2] = mshalo[i,ind2-1]
#                    mshalo[i,ind2] = mshalo[i,ind2-1]

        if i==0 or i==1 or i==2:
            for j in snap[i]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mhalo[i,:])
            ind = np.where(mhalo[i,:]>0)
            ax.plot(snap_list[ind],np.log10(mhalo[i,ind])[0],'.', c='darkgrey',linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkgrey', linewidth=linw1)
            prop = np.log10(mshalo[i,:])
            ind = np.where(mshalo[i,:]>0)
            ax.plot(snap_list[ind],np.log10(mshalo[i,ind])[0],'.', c='black',linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='black', linewidth=linw1)
        elif i==3 or i==4 or i==5:
            for j in snap[i-3]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-3,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-3,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(total_sfh_z0[i-3,:])
            ind = np.where(total_sfh_z0[i-3,:]>0)
            ax.plot(snap_list[ind],np.log10(total_sfh_z0[i-3,ind])[0],'.', c='blue', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='blue', linewidth=linw1)
            prop = np.log10(sfh_b[i-3,:])
            ind = np.where(sfh_b[i-3,:]>0)
            ax.plot(snap_list[ind],np.log10(sfh_b[i-3,ind])[0],'.', c='darkblue', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkblue', linewidth=linw1)
            prop = np.log10(sfh_c[i-3,:])
            ind = np.where(sfh_c[i-3,:]>0)
            ax.plot(snap_list[ind],np.log10(sfh_c[i-3,ind])[0],'.', c='lightblue', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='lightblue', linewidth=linw1)
            prop = np.log10(sfh_d[i-3,:])
            ind = np.where(sfh_d[i-3,:]>0)
            ax.plot(snap_list[ind],np.log10(sfh_d[i-3,ind])[0],'.', c='cyan', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='cyan', linewidth=linw1)
        elif i==6 or i==7 or i==8:
            for j in snap[i-6]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-6,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-6,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mdisk[i-6,:]+mbulge[i-6,:])
            ind = np.where((mdisk[i-6,:]+mbulge[i-6,:])>0)
            ax.plot(snap_list[ind],np.log10(mdisk[i-6,ind]+mbulge[i-6,ind])[0],'.', c='darkviolet', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkviolet', linewidth=linw1)
            prop = np.log10(mdisk[i-6,:])
            ind = np.where(mdisk[i-6,:]>0)
            ax.plot(snap_list[ind],np.log10(mdisk[i-6,ind])[0],'.', c='springgreen', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='springgreen', linewidth=linw1)
            prop = np.log10(mbulge[i-6,:])
            ind = np.where(mbulge[i-6,:]>0)
            ax.plot(snap_list[ind],np.log10(mbulge[i-6,ind])[0],'.', c='khaki', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='khaki', linewidth=linw1)
        elif i==9 or i==10 or i==11:
            for j in snap[i-9]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-9,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-9,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mhot[i-9,:])
            ind = np.where(mhot[i-9,:]>0)
            ax.plot(snap_list[ind],np.log10(mhot[i-9,ind])[0],'.', c='orange', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='orange', linewidth=linw1)
        elif i==12 or i==13 or i==14:
            for j in snap[i-12]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-12,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-12,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mgas_disk[i-12,:]+mgas_bulge[i-12,:])
            ind = np.where((mgas_disk[i-12,:]+mgas_bulge[i-12,:])>0)
            ax.plot(snap_list[ind],np.log10((mgas_disk[i-12,ind]+mgas_bulge[i-12,ind]))[0],'.', c='darkgreen', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkgreen', linewidth=linw1)
            prop = np.log10(mgas_disk[i-12,:])
            ind = np.where(mgas_disk[i-12,:]>0)
            ax.plot(snap_list[ind],np.log10(mgas_disk[i-12,ind])[0],'.', c='magenta', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='magenta', linewidth=linw1)
            prop = np.log10(mgas_bulge[i-12,:])
            ind = np.where(mgas_bulge[i-12,:]>0)
            ax.plot(snap_list[ind],np.log10(mgas_bulge[i-12,ind])[0],'.', c='maroon', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='maroon', linewidth=linw1)
        elif i==15 or i==16 or i==17:
            for j in snap[i-15]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-15,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-15,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(rgas_disk[i-15,:])
            ind = np.where(rgas_disk[i-15,:]>0)
            ax.plot(snap_list[ind],np.log10(rgas_disk[i-15,ind])[0],'.', c='gold', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='gold', linewidth=linw1)
        elif i==18 or i==19 or i==20:
            for j in snap[i-18]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-18,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-18,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(rgas_bulge[i-18,:])
            ind = np.where(rgas_bulge[i-18,:]>0)
            ax.plot(snap_list[ind],np.log10(rgas_bulge[i-18,ind])[0],'.', c='pink', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='pink', linewidth=linw1)
        elif i==21 or i==22 or i==23:
            for j in snap[i-21]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-21,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-21,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(bh_accretion_hh[i-21,:]+bh_accretion_sb[i-21,:])
            ind = np.where((bh_accretion_hh[i-21,:]+bh_accretion_sb[i-21,:])>0)
            ax.plot(snap_list[ind],np.log10(bh_accretion_hh[i-21,ind]+bh_accretion_sb[i-21,ind])[0],'.', c='teal', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='teal', linewidth=linw1)
            prop = np.log10(bh_accretion_hh[i-21,:])
            ind = np.where(bh_accretion_hh[i-21,:]>0)
            ax.plot(snap_list[ind],np.log10(bh_accretion_hh[i-21,ind])[0],'.', c='crimson', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='crimson', linewidth=linw1)
            prop = np.log10(bh_accretion_sb[i-21,:])
            ind = np.where(bh_accretion_sb[i-21,:]>0)
            ax.plot(snap_list[ind],np.log10(bh_accretion_sb[i-21,ind])[0],'.', c='palegreen', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='palegreen', linewidth=linw1)
        elif i==24 or i==25 or i==26:
            for j in snap[i-24]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-24,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-24,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mbh[i-24,:])
            ind = np.where(mbh[i-24,:]>0)
            ax.plot(snap_list[ind],np.log10(mbh[i-24,ind])[0],'.', c='olive', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='olive', linewidth=linw1)
            prop = np.log10(mbh_accreted[i-24,:])
            ind = np.where(mbh_accreted[i-24,:]>0)
            ax.plot(snap_list[ind],np.log10(mbh_accreted[i-24,ind])[0],'.', c='seagreen', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='seagreen', linewidth=linw1)
        elif i==27 or i==28 or i==29:
            for j in snap[i-27]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-27,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-27,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(matom[i-27,:])
            ind = np.where(matom[i-27,:]>0)
            ax.plot(snap_list[ind],np.log10(matom[i-27,ind])[0],'.', c='purple', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='purple', linewidth=linw1)
            prop = np.log10(matom_disk[i-27,:])
            ind = np.where(matom_disk[i-27,:]>0)
            ax.plot(snap_list[ind],np.log10(matom_disk[i-27,ind])[0],'.', c='lime', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='lime', linewidth=linw1)
            prop = np.log10(matom_bulge[i-27,:])
            ind = np.where(matom_bulge[i-27,:]>0)
            ax.plot(snap_list[ind],np.log10(matom_bulge[i-27,ind])[0],'.', c='peru', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='peru', linewidth=linw1)
        elif i==30 or i==31 or i==32:
            for j in snap[i-30]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-30,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-30,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mmol[i-30,:])
            ind = np.where(mmol[i-30,:]>0)
            ax.plot(snap_list[ind],np.log10(mmol[i-30,ind])[0],'.', c='turquoise', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='turquoise', linewidth=linw1)
            prop = np.log10(mmol_disk[i-30,:])
            ind = np.where(mmol_disk[i-30,:]>0)
            ax.plot(snap_list[ind],np.log10(mmol_disk[i-30,ind])[0],'.', c='orchid', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='orchid', linewidth=linw1)
            prop = np.log10(mmol_bulge[i-30,:])
            ind = np.where(mmol_bulge[i-30,:]>0)
            ax.plot(snap_list[ind],np.log10(mmol_bulge[i-30,ind])[0],'.', c='salmon', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='salmon', linewidth=linw1)
        elif i==33 or i==34 or i==35:
            for j in snap[i-33]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-33,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-33,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mreheated[i-33,:])
            ind = np.where(mreheated[i-33,:]>0)
            ax.plot(snap_list[ind],np.log10(mreheated[i-33,ind])[0],'.', c='plum', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='plum', linewidth=linw1)
            prop = np.log10(mlost[i-33,:])
            ind = np.where(mlost[i-33,:]>0)
            ax.plot(snap_list[ind],np.log10(mlost[i-33,ind])[0],'.', c='sienna', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='sienna', linewidth=linw1)
        elif i==36 or i==37 or i==38:
            for j in snap[i-36]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-36,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-36,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(vvir_h[i-36,:])
            ind = np.where(vvir_h[i-36,:]>0)
            ax.plot(snap_list[ind],np.log10(vvir_h[i-36,ind])[0],'.', c='darkred', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkred', linewidth=linw1)
            prop = np.log10(vvir_sh[i-36,:])
            ind = np.where(vvir_sh[i-36,:]>0)
            ax.plot(snap_list[ind],np.log10(vvir_sh[i-36,ind])[0],'.', c='tomato', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='tomato', linewidth=linw1)
            prop = np.log10(vmax_sh[i-36,:])
            ind = np.where(vmax_sh[i-36,:]>0)
            ax.plot(snap_list[ind],np.log10(vmax_sh[i-36,ind])[0],'.', c='lightsalmon', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='lightsalmon', linewidth=linw1)

        if i==0:
            handles1,labels = plt.gca().get_legend_handles_labels()
            patch0 = mpatches.Patch(color='darkgrey', label='Halo mass')
            patch1 = mpatches.Patch(color='black', label='Subhalo mass')
            patch2 = mpatches.Patch(color='blue', label='SHARK SFR')
            patch3 = mpatches.Patch(color='darkblue', label='SFR bulge disk-instabilities')
            patch4 = mpatches.Patch(color='lightblue', label='SFR bulge mergers')
            patch5 = mpatches.Patch(color='cyan', label='SFR disk')
            patch6 = mpatches.Patch(color='darkviolet', label='Stellar mass')
            patch7 = mpatches.Patch(color='springgreen', label='Stellar disk mass')
            patch8 = mpatches.Patch(color='khaki', label='Stellar bulge mass')
            patch9 = mpatches.Patch(color='orange', label='Hot gas')
            patch10 = mpatches.Patch(color='darkgreen', label='Cold gas mass')
            patch11 = mpatches.Patch(color='magenta', label='Cold gas disk mass')
            patch12 = mpatches.Patch(color='maroon', label='Cold gas bulge mass')
            patch13 = mpatches.Patch(color='gold', label='Cold gas disk size')
            patch14 = mpatches.Patch(color='pink', label='Cold gas bulge size')
            patch15 = mpatches.Patch(color='teal', label='BH accretion')
            patch16 = mpatches.Patch(color='crimson', label='BH hot halo accretion')
            patch17 = mpatches.Patch(color='palegreen', label='BH sb accretion')
            patch18 = mpatches.Patch(color='olive', label='BH mass')
            patch19 = mpatches.Patch(color='seagreen', label='BH accreted mass')
            patch20 = mpatches.Patch(color='purple', label='HI mass')
            patch21 = mpatches.Patch(color='lime', label='HI disk mass')
            patch22 = mpatches.Patch(color='peru', label='HI bulge mass')
            patch23 = mpatches.Patch(color='turquoise', label='H2 mass')
            patch24 = mpatches.Patch(color='orchid', label='H2 disk mass')
            patch25 = mpatches.Patch(color='salmon', label='H2 bulge mass')
            patch26 = mpatches.Patch(color='plum', label='Reheated mass')
            patch27 = mpatches.Patch(color='sienna', label='Lost mass')
            patch28 = mpatches.Patch(color='darkred', label='Halo vir velocity')
            patch29 = mpatches.Patch(color='tomato', label='Subhalo vir velocity')
            patch30 = mpatches.Patch(color='lightsalmon', label='Max velocity')
#            line1 = Line2D([0],[0],label='$\Delta\mathrm{M}/\mathrm{M_{i}}<-0.9\ \mathrm{mass}\ \mathrm{drop}$',linestyle="--",color='red',linewidth=linw2)
#            line2 = Line2D([0],[0],label='$\mathrm{N}_{\mathrm{part,born}}>10^{3}$',linestyle="--",color='green',linewidth=linw2)
            line1 = Line2D([0],[0],label='Mass swapping',linestyle="--",color='red',linewidth=linw2)
            line2 = Line2D([0],[0],label='$\mathrm{N}_{\mathrm{part,born}}>10^{2}$',linestyle=":",color='green',linewidth=linw2)
            line3 = Line2D([0],[0],label='$\mathrm{N}_{\mathrm{part,born}}>10^{3}$',linestyle="--",color='green',linewidth=linw2)
            handles1.extend([patch0,patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10,patch11,patch12,patch13,patch14,patch15,patch16,patch17,patch18,patch19,patch20,patch21,patch22,patch23,patch24,patch25,patch26,patch27,patch28,patch29,patch30,line1,line2,line3])
            fig.legend(handles=handles1,loc='lower center',frameon=True,framealpha=1,bbox_to_anchor=(1.1, 0.5))

    common.savefig(outdir, fig, "MAHs-snap_wrongbranch_all.png")


def prepare_data_nosfh(hdf5_data, index):
   
    #star_formation_histories and SharkSED have the same number of galaxies in the same order, and so we can safely assume that to be the case.
    #to select the same galaxies in galaxies.hdf5 we need to ask for all of those that have a stellar mass > 0, and then assume that they are in the same order.

    (h0, _, mdisk, mbulge, mhalo, mshalo, typeg, age,
     sfr_disk, sfr_burst, id_gal, mbh, mhot, mreheated,
     id_subhalo, id_halo, matom_bulge, matom_disk, mmol_bulge,
     mmol_disk, mstellar_halo, mgas_bulge, mgas_disk,
     mstars_stripped, rhalo_stripped, rism_stripped,
     rstar_bulge, rstar_disk, rgas_bulge, rgas_disk,
     bh_accretion_hh, bh_accretion_sb, mbh_assembly,
     mlost, sfr_burst_diskins, sfr_burst_mergers,
     vmax_sh, vvir_h, vvir_sh) = hdf5_data

    sfr_tot = (sfr_disk + sfr_burst)/1e9/h0

    #components:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    #ignore last band which is the top-hat UV of high-z LFs.
    ind = np.where(mdisk + mbulge > 0)
    ngals       = len(mdisk[ind])
    gal_props = np.zeros(shape = (ngals, 42))

    gal_props[:,0] = 13.6-age[ind]
    gal_props[:,1] = mdisk[ind] + mbulge[ind]
    gal_props[:,2] = mbulge[ind] / (mdisk[ind] + mbulge[ind])
    gal_props[:,3] = (sfr_burst[ind] + sfr_disk[ind])/1e9/h0
    gal_props[:,4] = typeg[ind]
    gal_props[:,5] = id_subhalo[ind]
    gal_props[:,6] = id_halo[ind]
    gal_props[:,7] = mbh[ind]
    gal_props[:,8] = mgas_bulge[ind] + mgas_disk[ind]
    gal_props[:,9] = mshalo[ind]
    gal_props[:,10] = mhalo[ind]    
    gal_props[:,11] = mstellar_halo[ind]
    gal_props[:,12] = mmol_bulge[ind] + mmol_disk[ind]   
    gal_props[:,13] = matom_bulge[ind] + matom_disk[ind]   
    gal_props[:,14] = mhot[ind]   
    gal_props[:,15] = mbulge[ind]   
    gal_props[:,16] = mdisk[ind]   
    gal_props[:,17] = mgas_bulge[ind]   
    gal_props[:,18] = mgas_disk[ind]   
    gal_props[:,19] = mstars_stripped[ind]   
    gal_props[:,20] = rhalo_stripped[ind]   
    gal_props[:,21] = rism_stripped[ind]   
    gal_props[:,22] = rstar_bulge[ind]   
    gal_props[:,23] = rstar_disk[ind]   
    gal_props[:,24] = rgas_bulge[ind]   
    gal_props[:,25] = rgas_disk[ind]   
    gal_props[:,26] = id_gal[ind]
    gal_props[:,27] = bh_accretion_hh[ind]/1e9/h0  
    gal_props[:,28] = bh_accretion_sb[ind]/1e9/h0   
    gal_props[:,29] = mbh[ind] - mbh_assembly[ind]  
    gal_props[:,30] = mlost[ind]  
    gal_props[:,31] = mreheated[ind]
    gal_props[:,32] = (sfr_disk[ind])/1e9/h0
    gal_props[:,33] = (sfr_burst_diskins[ind])/1e9/h0
    gal_props[:,34] = (sfr_burst_mergers[ind])/1e9/h0
    gal_props[:,35] = mmol_bulge[ind]
    gal_props[:,36] = mmol_disk[ind]   
    gal_props[:,37] = matom_bulge[ind]
    gal_props[:,38] = matom_disk[ind]  
    gal_props[:,39] = vmax_sh[ind]   
    gal_props[:,40] = vvir_h[ind]
    gal_props[:,41] = vvir_sh[ind]   

  
    return (gal_props)


def main(model_dir, outdir, redshift_table, subvols, obsdir):


    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    fields_nosfh = {'galaxies': ('mstars_disk', 'mstars_bulge', 'mvir_hosthalo',
                                 'mvir_subhalo', 'type', 'mean_stellar_age',
                                 'sfr_disk', 'sfr_burst', 'id_galaxy', 'm_bh',
                                 'mhot','mreheated','id_subhalo_tree','id_halo_tree',
                                 'matom_bulge', 'matom_disk', 'mmol_bulge',
                                 'mmol_disk', 'mstellar_halo', 'mgas_bulge',
                                 'mgas_disk', 'mgas_disk', 'mstars_tidally_stripped', 'r_halo_stripped',
                                 'r_ism_stripped', 'rstar_bulge', 'rstar_disk',
                                 'rgas_bulge', 'rgas_disk', 'bh_accretion_rate_hh',
                                 'bh_accretion_rate_sb','m_bh_assembly',
                                 'mlost','sfr_burst_diskins','sfr_burst_mergers',
                                 'vmax_subhalo','vvir_hosthalo','vvir_subhalo')}
    fields_test = {'subhalo': ('id','descendant_id', 'main_progenitor','host_id')}
    
    sfh_fields = {'bulges_diskins': ('star_formation_rate_histories'),
                  'bulges_mergers': ('star_formation_rate_histories'),
                  'disks': ('star_formation_rate_histories'),
                  'galaxies': ('id_galaxy')}

    snapshots = np.arange(1,sim_dict['snapshots'],1)
    
    # Create histogram

    galaxyID = np.array([],dtype='int32')

    mstars = np.zeros(shape=(len(ID),len(snapshots)))
    mbh = np.zeros(shape=(len(ID),len(snapshots)))
    mgas = np.zeros(shape=(len(ID),len(snapshots)))
    mshalo = np.zeros(shape=(len(ID),len(snapshots)))
    mhalo = np.zeros(shape=(len(ID),len(snapshots)))
    mstellar_halo = np.zeros(shape=(len(ID),len(snapshots)))
    mmol = np.zeros(shape=(len(ID),len(snapshots)))
    matom = np.zeros(shape=(len(ID),len(snapshots)))
    mhot = np.zeros(shape=(len(ID),len(snapshots)))
    mbulge = np.zeros(shape=(len(ID),len(snapshots)))
    mdisk = np.zeros(shape=(len(ID),len(snapshots)))
    mgas_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    mgas_disk = np.zeros(shape=(len(ID),len(snapshots)))
    mstars_stripped = np.zeros(shape=(len(ID),len(snapshots)))
    rhalo_stripped = np.zeros(shape=(len(ID),len(snapshots)))
    rism_stripped = np.zeros(shape=(len(ID),len(snapshots)))
    rbulge = np.zeros(shape=(len(ID),len(snapshots)))
    rdisk = np.zeros(shape=(len(ID),len(snapshots)))
    rgas_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    rgas_disk = np.zeros(shape=(len(ID),len(snapshots)))
    sfh_a = np.zeros(shape=(len(ID),len(snapshots)))
    sfh_b = np.zeros(shape=(len(ID),len(snapshots)))
    sfh_c = np.zeros(shape=(len(ID),len(snapshots)))
    sfh_d = np.zeros(shape=(len(ID),len(snapshots)))
    bh_accretion_hh = np.zeros(shape=(len(ID),len(snapshots)))
    bh_accretion_sb = np.zeros(shape=(len(ID),len(snapshots)))
    mbh_accreted = np.zeros(shape=(len(ID),len(snapshots)))
    mean_stellar_age = np.zeros(shape=(len(ID),len(snapshots)))
    mlost = np.zeros(shape=(len(ID),len(snapshots)))
    mreheated = np.zeros(shape=(len(ID),len(snapshots)))
    sfr = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_disk = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_burst_diskins = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_burst_mergers = np.zeros(shape=(len(ID),len(snapshots)))
    matom_disk = np.zeros(shape=(len(ID),len(snapshots)))
    matom_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    mmol_disk = np.zeros(shape=(len(ID),len(snapshots)))
    mmol_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    vmax_sh = np.zeros(shape=(len(ID),len(snapshots)))
    vvir_sh = np.zeros(shape=(len(ID),len(snapshots)))
    vvir_h = np.zeros(shape=(len(ID),len(snapshots)))

    # find central galaxy
    index = 0
    hdf5_data_nosfh = common.read_data(model_dir, snapshots[-1], fields_nosfh, subvols)
    (gal_props) = prepare_data_nosfh(hdf5_data_nosfh, index)
    h0, volh = hdf5_data_nosfh[0], hdf5_data_nosfh[1]
    gal_props_z0 = gal_props

    hdf5_data_test = common.read_data(model_dir, snapshots[-1], fields_test, subvols)
#    ( _, _, sid ,descendant_id, main_progenitor, host_id) = hdf5_data_test
#    ind_test = np.where(sid==70000064830064)
#    did = descendant_id[ind_test]
#    print("HERE:",did)
#    while did!=-1:
#        ind = np.where(sid==did)
#        did = descendant_id[ind]
#        print(did)
#    ind_test = np.where((descendant_id==sid)&(main_progenitor==1))
#    mid = sid[ind_test]
#    print("THERE:",mid)
#    while mid!=-1:
#        ind = np.where((descendant_id==mid)&(main_progenitor==1))
#        mid = sid[ind]
#        print(mid)
    
    # know which subvolume we have to consider
    nsubv = 1
    # read files
    plot_dir = "/fred/oz009/achandro/track_halos/"+sim_dict["sim"]+"/"
    f = h5py.File(plot_dir+'mainbranch_'+sim_dict["sim"]+'.0.hdf5','r')
    subhaloID = f["/data/nodeIndex"][()]
    snap = f["/data/snapshot"][()]
    finalID_raw = f["/data/nodeIndex_finalsnap"][()]
    for i in range(len(ID)):
        ind_h = np.where((snap==snapshots[-1])&(finalID_raw==ID[i]))
        if np.shape(ind_h)[1]>0:
            sID = subhaloID[ind_h]
            ind = np.where((gal_props_z0[:,5] == sID) & (gal_props_z0[:,4] == 0))
            galaxyID = np.append(galaxyID,int(gal_props_z0[ind,26]))
    print("subhaloID:",sID)
    print("galaxyID:",galaxyID)
            

    
    for index, snapshot in enumerate(snapshots):

        hdf5_data_nosfh = common.read_data(model_dir, snapshot, fields_nosfh, subvols)
        if snapshot==snapshots[-1]:
            sfh, delta_t, LBT, subv_a = read_sfh_subv(model_dir, snapshot, sfh_fields, subvols)

#            (total_sfh, sb_sfh, sbd_sfh, sbm_sfh, disk_sfh, gal_props, id_gal_sfh) = prepare_data(hdf5_data, sfh, index)           
#            total_sfh_z0 = total_sfh
#            sbd_sfh_z0 = sbd_sfh
#            sbm_sfh_z0 = sbm_sfh
#            disk_sfh_z0 = disk_sfh
#            gal_props_z0 = gal_props
            LBT_z0 = LBT

            (gal_props) = prepare_data_nosfh(hdf5_data_nosfh, index)
            h0, volh = hdf5_data_nosfh[0], hdf5_data_nosfh[1]
            gal_props_z0 = gal_props


        else:
            
            (gal_props) = prepare_data_nosfh(hdf5_data_nosfh, index)
            h0, volh = hdf5_data_nosfh[0], hdf5_data_nosfh[1]
            gal_props_z0 = gal_props

#        # know which subvolume we have to consider
#        nsubv = 1

        # read files
        plot_dir = "/fred/oz009/achandro/track_halos/"+sim_dict["sim"]+"/"
        f = h5py.File(plot_dir+'mainbranch_'+sim_dict["sim"]+'.0.hdf5','r')
        subhaloID = f["/data/nodeIndex"][()]
        snap = f["/data/snapshot"][()]
        finalID_raw = f["/data/nodeIndex_finalsnap"][()]

        print("")
        print("snaphot:",snapshot)
#        ind_h = np.where((snap==snapshot)&(finalID_raw==ID[0]))
#        if np.shape(ind_h)[1]>0:
#            sID = subhaloID[ind_h]
#        indgal = np.where(gal_props_z0[:,26]==galaxyID[0])
#        if np.shape(indgal)[1]>0 and np.shape(ind_h)[1]>0:
#            if np.array(gal_props_z0[indgal,5],dtype='int64')[0][0]!=sID[0] and snapshot>=snap[np.where(finalID_raw==ID[0])][0]:
#                print("i:0")
#                print("Different subhaloID: "+str(sID[0])+" (subhaloID for MB), "+str(np.array(gal_props_z0[indgal,5],dtype='int64')[0][0])+" (subhaloID for central galaxy)")
#                print("haloID0:",np.array(gal_props_z0[indgal,6],dtype='int64')[0][0])
#                print("type0:",gal_props_z0[indgal,4][0][0])
#
#        ind_h = np.where((snap==snapshot)&(finalID_raw==ID[1]))
#        if np.shape(ind_h)[1]>0:
#            sID = subhaloID[ind_h]
#        indgal = np.where(gal_props_z0[:,26]==galaxyID[1])
#        if np.shape(indgal)[1]>0 and np.shape(ind_h)[1]>0 and snapshot>=snap[np.where(finalID_raw==ID[0])][0]:
#            if np.array(gal_props_z0[indgal,5],dtype='int64')[0][0]!=sID[0]:
#                print("i:1")
#                print("Different subhaloID: "+str(sID[0])+" (subhaloID for MB), "+str(np.array(gal_props_z0[indgal,5],dtype='int64')[0][0])+" (subhaloID for central galaxy)")
#                print("haloID0:",np.array(gal_props_z0[indgal,6],dtype='int64')[0][0])
#                print("type0:",gal_props_z0[indgal,4][0][0])
#
        ind_h = np.where((snap==snapshot)&(finalID_raw==ID[2]))
        print(snap[np.where(finalID_raw==ID[2])])
        if np.shape(ind_h)[1]>0:
            sID = subhaloID[ind_h]
            print(sID)
        indgal = np.where(gal_props_z0[:,26]==galaxyID[2])
        print(indgal)
        print("subhaloID: ",np.array(gal_props_z0[indgal,5][0],dtype='int64'))
        indtest = np.where((gal_props_z0[:,6] == gal_props_z0[indgal,5][0]))
        if np.shape(indtest)[1]>0:
            print("ih:2")
            print("type:",gal_props_z0[indtest,4])
            print("subhaloID:",np.array(gal_props_z0[indtest,5],dtype='int64'))
            print("haloID:",np.array(gal_props_z0[indtest,6],dtype='int64'))
            print("galaxyID:",gal_props_z0[indtest,26])
            print("mbulge_gas:",gal_props_z0[indtest,17])
            print("mbulge_stars:",gal_props_z0[indtest,15])
            print("mdisk_stars:",gal_props_z0[indtest,16])
            print("mlost:",gal_props_z0[indtest,30])
            print("mgalaxy:",gal_props_z0[indtest,15]+gal_props_z0[indtest,16]+gal_props_z0[indtest,17]+gal_props_z0[indtest,18])
        else:
            indtest = np.where((gal_props_z0[:,5] == gal_props_z0[indgal,5][0]))
            if np.shape(indtest)[1]>0:
                print("ish:2")
                print("type:",gal_props_z0[indtest,4])
                print("subhaloID:",np.array(gal_props_z0[indtest,5],dtype='int64'))
                print("haloID:",np.array(gal_props_z0[indtest,6],dtype='int64'))
                print("galaxyID:",gal_props_z0[indtest,26])
                print("mbulge_gas:",gal_props_z0[indtest,17])
                print("mbulge_stars:",gal_props_z0[indtest,15])
                print("mdisk_stars:",gal_props_z0[indtest,16])
                print("mlost:",gal_props_z0[indtest,30])
                print("mgalaxy:",gal_props_z0[indtest,15]+gal_props_z0[indtest,16]+gal_props_z0[indtest,17]+gal_props_z0[indtest,18])
#        if np.shape(indgal)[1]>0 and np.shape(ind_h)[1]>0:
#            if np.array(gal_props_z0[indgal,5],dtype='int64')[0][0]!=sID[0] and snapshot>=snap[np.where(finalID_raw==ID[0])][0]:
#                print("i:2")
#                print("Different subhaloID: "+str(sID[0])+" (subhaloID for MB), "+str(np.array(gal_props_z0[indgal,5],dtype='int64')[0][0])+" (subhaloID for central galaxy)")
#                print("haloID0:",np.array(gal_props_z0[indgal,6],dtype='int64')[0][0])
#                print("type0:",gal_props_z0[indgal,4][0][0])

        for i in range(len(ID)):

            ind = np.where((gal_props_z0[:,26] == galaxyID[i]))
            if np.shape(ind)[1]>0:
                mstars[i,index] = gal_props_z0[ind,1]
                mbh[i,index] = gal_props_z0[ind,7]
                mgas[i,index] = gal_props_z0[ind,8]
                mshalo[i,index] = gal_props_z0[ind,9]
                mhalo[i,index] = gal_props_z0[ind,10]
                mstellar_halo[i,index] = gal_props_z0[ind,11]
                mmol[i,index] = gal_props_z0[ind,12]
                matom[i,index] = gal_props_z0[ind,13]
                mhot[i,index] = gal_props_z0[ind,14]
                mbulge[i,index] = gal_props_z0[ind,15]
                mdisk[i,index] = gal_props_z0[ind,16]
                mgas_bulge[i,index] = gal_props_z0[ind,17]
                mgas_disk[i,index] = gal_props_z0[ind,18]
                mstars_stripped[i,index] = gal_props_z0[ind,19]
                rhalo_stripped[i,index] = gal_props_z0[ind,20]
                rism_stripped[i,index] = gal_props_z0[ind,21]
                rbulge[i,index] = gal_props_z0[ind,22]
                rdisk[i,index] = gal_props_z0[ind,23]
                rgas_bulge[i,index] = gal_props_z0[ind,24]
                rgas_disk[i,index] = gal_props_z0[ind,25]
                bh_accretion_hh[i,index] = gal_props_z0[ind,27]
                bh_accretion_sb[i,index] = gal_props_z0[ind,28]
                mbh_accreted[i,index] = gal_props_z0[ind,29]
                mean_stellar_age[i,index] = gal_props_z0[ind,0]
                mlost[i,index] = gal_props_z0[ind,30]
                mreheated[i,index] = gal_props_z0[ind,31]
                sfr[i,index] = gal_props_z0[ind,32]+gal_props_z0[ind,33]+gal_props_z0[ind,34]
                sfr_disk[i,index] = gal_props_z0[ind,32]
                sfr_burst_diskins[i,index] = gal_props_z0[ind,33]
                sfr_burst_mergers[i,index] = gal_props_z0[ind,34]
                mmol_bulge[i,index] = gal_props_z0[ind,35]
                mmol_disk[i,index] = gal_props_z0[ind,36]
                matom_bulge[i,index] = gal_props_z0[ind,37]
                matom_disk[i,index] = gal_props_z0[ind,38]
                vmax_sh[i,index] = gal_props_z0[ind,39]
                vvir_h[i,index] = gal_props_z0[ind,40]
                vvir_sh[i,index] = gal_props_z0[ind,41]
#                if snapshot==snapshots[-1]:
#                    for isub in range(nsubv):
#                        dh_file = sim_dict["dhalo_dir"]+"/tree_078."+str(isub)+".hdf5"
#                        with h5py.File(dh_file, 'r') as f:
#                            if np.shape(np.where(f['haloTrees/nodeIndex'][()]==ID[i]))[1]>0:
#                                subv = isub
#                                print("Subvolume: ",subv)
#                    # associated id_galaxy:
#                    ind_sfh = np.where((id_gal_sfh==galaxyID[i]))
#                    tot_sfh_selec = total_sfh_z0[ind_sfh,:]
#                    tot_sfh_selec = tot_sfh_selec[0,:]
#                    sfh_a[i,:] = tot_sfh_selec
#                    sbd_sfh_selec = sbd_sfh_z0[ind_sfh,:]
#                    sbd_sfh_selec = sbd_sfh_selec[0,:]
#                    sfh_b[i,:] = sbd_sfh_selec
#                    sbm_sfh_selec = sbm_sfh_z0[ind_sfh,:]
#                    sbm_sfh_selec = sbm_sfh_selec[0,:]
#                    sfh_c[i,:] = sbm_sfh_selec
#                    disk_sfh_selec = disk_sfh_z0[ind_sfh,:]
#                    disk_sfh_selec = disk_sfh_selec[0,:]
#                    sfh_d[i,:] = disk_sfh_selec
#                    
#            ind_h = np.where((snap==snapshot)&(finalID_raw==ID[i]))
#            if np.shape(ind_h)[1]>0:
#                sID = subhaloID[ind_h]
#                ind = np.where((gal_props_z0[:,5] == sID) & (gal_props_z0[:,4] == 0))
#                print("subhaloID:",sID)
#                indtest = np.where((gal_props_z0[:,6] == sID))
#                if i==0 and np.shape(indtest)[1]>0:
#                    print("type:",gal_props_z0[indtest,4])
#                    print("subhaloID:",np.array(gal_props_z0[indtest,5],dtype='int64'))
#                    print("haloID:",np.array(gal_props_z0[indtest,6],dtype='int64'))
#                    print("galaxyID:",gal_props_z0[indtest,26])
#                    print("mbulge_gas:",gal_props_z0[indtest,17])
#                    print("mgalaxy:",gal_props_z0[indtest,15]+gal_props_z0[indtest,16]+gal_props_z0[indtest,17]+gal_props_z0[indtest,18])
#                if snapshot==snapshots[-1] and np.shape(indtest)[1]>0:
#                    print('i:',i)
#                    print("type:",gal_props_z0[indtest,4])
#                    print("subhaloID:",np.array(gal_props_z0[indtest,5],dtype='int64'))
#                    print("haloID:",np.array(gal_props_z0[indtest,6],dtype='int64'))
#                    print("galaxyID:",gal_props_z0[indtest,26])
#                    print("mbulge_gas:",gal_props_z0[indtest,17])
#                    print("mgalaxy:",gal_props_z0[indtest,15]+gal_props_z0[indtest,16]+gal_props_z0[indtest,17]+gal_props_z0[indtest,18])
#                if np.shape(ind)[1]>0:
#                    mstars[i,index] = gal_props_z0[ind,1]
#                    mbh[i,index] = gal_props_z0[ind,7]
#                    mgas[i,index] = gal_props_z0[ind,8]
#                    mshalo[i,index] = gal_props_z0[ind,9]
#                    mhalo[i,index] = gal_props_z0[ind,10]
#                    mstellar_halo[i,index] = gal_props_z0[ind,11]
#                    mmol[i,index] = gal_props_z0[ind,12]
#                    matom[i,index] = gal_props_z0[ind,13]
#                    mhot[i,index] = gal_props_z0[ind,14]
#                    mbulge[i,index] = gal_props_z0[ind,15]
#                    mdisk[i,index] = gal_props_z0[ind,16]
#                    mgas_bulge[i,index] = gal_props_z0[ind,17]
#                    mgas_disk[i,index] = gal_props_z0[ind,18]
#                    mstars_stripped[i,index] = gal_props_z0[ind,19]
#                    rhalo_stripped[i,index] = gal_props_z0[ind,20]
#                    rism_stripped[i,index] = gal_props_z0[ind,21]
#                    rbulge[i,index] = gal_props_z0[ind,22]
#                    rdisk[i,index] = gal_props_z0[ind,23]
#                    rgas_bulge[i,index] = gal_props_z0[ind,24]
#                    rgas_disk[i,index] = gal_props_z0[ind,25]
#                    bh_accretion_hh[i,index] = gal_props_z0[ind,27]
#                    bh_accretion_sb[i,index] = gal_props_z0[ind,28]
#                    mbh_accreted[i,index] = gal_props_z0[ind,29]
#                    mean_stellar_age[i,index] = gal_props_z0[ind,0]
#                    mlost[i,index] = gal_props_z0[ind,30]
#                    mreheated[i,index] = gal_props_z0[ind,31]
#                    sfr[i,index] = gal_props_z0[ind,32]+gal_props_z0[ind,33]+gal_props_z0[ind,34]
#                    sfr_disk[i,index] = gal_props_z0[ind,32]
#                    sfr_burst_diskins[i,index] = gal_props_z0[ind,33]
#                    sfr_burst_mergers[i,index] = gal_props_z0[ind,34]
#                    if snapshot==snapshots[-1]:
#                        for isub in range(nsubv):
#                            dh_file = sim_dict["dhalo_dir"]+"/tree_078."+str(isub)+".hdf5"
#                            with h5py.File(dh_file, 'r') as f:
#                                if np.shape(np.where(f['haloTrees/nodeIndex'][()]==ID[i]))[1]>0:
#                                    subv = isub
#                                    print("Subvolume: ",subv)
#                        # associated id_galaxy:
#                        ind_sfh = np.where((id_gal_sfh==int(gal_props_z0[ind,26][0][0]))&(subv_a==subv))
#                        tot_sfh_selec = total_sfh_z0[ind_sfh,:]
#                        tot_sfh_selec = tot_sfh_selec[0,:]
#                        sfh_a[i,:] = tot_sfh_selec
#                        sbd_sfh_selec = sbd_sfh_z0[ind_sfh,:]
#                        sbd_sfh_selec = sbd_sfh_selec[0,:]
#                        sfh_b[i,:] = sbd_sfh_selec
#                        sbm_sfh_selec = sbm_sfh_z0[ind_sfh,:]
#                        sbm_sfh_selec = sbm_sfh_selec[0,:]
#                        sfh_c[i,:] = sbm_sfh_selec
#                        disk_sfh_selec = disk_sfh_z0[ind_sfh,:]
#                        disk_sfh_selec = disk_sfh_selec[0,:]
#                        sfh_d[i,:] = disk_sfh_selec


    plot_mah_snap(plt, outdir, obsdir, h0, mshalo, mhalo, mbh, mgas_disk, mhot, rgas_disk, bh_accretion_hh, bh_accretion_sb, matom, mmol, mbh_accreted, mean_stellar_age, mlost, mreheated, mgas_bulge, rgas_bulge, gal_props_z0, LBT, str(0), sfr, sfr_burst_diskins, sfr_burst_mergers, sfr_disk, mdisk, mbulge, matom_bulge, matom_disk, mmol_bulge, mmol_disk, vmax_sh, vvir_h, vvir_sh)
#    plot_mah_snap(plt, outdir, obsdir, h0, mshalo, mhalo, mbh, mgas_disk, mhot, rgas_disk, bh_accretion_hh, bh_accretion_sb, matom, mmol, mbh_accreted, mean_stellar_age, mlost, mreheated, mgas_bulge, rgas_bulge, gal_props_z0, LBT, str(0), sfh_a, sfh_b, sfh_c, sfh_d)
  
if __name__ == '__main__':
    main(*common.parse_args())
