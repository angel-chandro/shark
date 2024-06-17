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
#snap = [[50],[39],[55]]

# FINAL MT
ID = [78000058463701,78000018965657,78000049808992]
snap = [[62],[43],[47]]


simu = "FLAMINGO"
sim_dict = choose_sim(simu)
snap_list = np.arange(1,sim_dict['snapshots'],1)

def read_data_galform(model_dir, snapshot, fields, subvolumes, include_h0_volh=True):
    """Read the galaxies.hdf5 file for the given model/iz+snapshot/ivol+subvolume"""
    
    data = collections.OrderedDict()
    for idx, subv in enumerate(subvolumes):
        
        fname = os.path.join(model_dir, 'iz'+str(snapshot), 'ivol'+str(subv), 'galaxies.hdf5')
        logger.info('Reading galaxies data from %s', fname)
        with h5py.File(fname, 'r') as f:
            if idx == 0 and include_h0_volh:
                data['h0'] = f['Parameters/h0'][()]
                data['vol'] = f['Parameters/volume'][()] * len(subvolumes)
                
            for gname, dsnames in fields.items():
                group = f[gname]
                for dsname in dsnames:
                    full_name = '%s/%s' % (gname, dsname)
                    l = data.get(full_name, None)
                    if l is None:
                        l = group[dsname][()]
                    else:
                        l = np.concatenate([l, group[dsname][()]])
                    data[full_name] = l
                        
    return list(data.values())

def plot_mah_snap(plt, outdir, h0, mshalo, mhalo, mbh, mstars_disk, mstars_bulge, mhot, mcold, rdisk, rbulge, bh_accretion_hh, bh_accretion_sb, matom_disk, matom_bulge, mmol_disk, mmol_bulge, gal_props_z0, LBT, redshift, sfr_burst, sfr_disk, vchalo, vhhalo):

    xtit="Snapshot"
    ytit="$\\rm log_{10}(M_{subhalo}/(M_{\odot}/h))$"
    ytit2="$\\rm log_{10}(SFR/M_{\odot} yr^{-1})$"
    ytit3="$\\rm log_{10}(M_{stars}/(M_{\odot}/h))$"
    ytit4="$\\rm log_{10}(M_{hot}/(M_{\odot}/h))$"
    ytit5="$\\rm log_{10}(M_{gas}/(M_{\odot}/h))$"
    ytit6="$\\rm log_{10}(R_{disk}/(Mpc/h))$"
    ytit7="$\\rm log_{10}(R_{bulge}/(Mpc/h))$"
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
            ax.plot(snap_list[ind],np.log10(mhalo[i,ind])[0],'.', c='darkgrey', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkgrey', linewidth=linw1)
            prop = np.log10(mshalo[i,:])
            ind = np.where(mshalo[i,:]>0)
            ax.plot(snap_list[ind],np.log10(mshalo[i,ind])[0],'.', c='black', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='black', linewidth=linw1)
        elif i==3 or i==4 or i==5:
            for j in snap[i-3]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-3,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-3,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(sfr_disk[i-3,:]+sfr_burst[i-3,:])
            ind = np.where((sfr_disk[i-3,:]+sfr_burst[i-3,:])>0)
            ax.plot(snap_list[ind],np.log10(sfr_disk[i-3,ind]+sfr_burst[i-3,ind])[0],'.', c='blue', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='blue', linewidth=linw1)
            prop = np.log10(sfr_burst[i-3,:])
            ind = np.where(sfr_burst[i-3,:]>0)
            ax.plot(snap_list[ind],np.log10(sfr_burst[i-3,ind])[0],'.', c='lightblue', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='lightblue', linewidth=linw1)
            prop = np.log10(sfr_disk[i-3,:])
            ind = np.where(sfr_disk[i-3,:]>0)
            ax.plot(snap_list[ind],np.log10(sfr_disk[i-3,ind])[0],'.', c='cyan', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='cyan', linewidth=linw1)
        elif i==6 or i==7 or i==8:
            for j in snap[i-6]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-6,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-6,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(mstars_disk[i-6,:]+mstars_bulge[i-6,:])
            ind = np.where((mstars_disk[i-6,:]+mstars_bulge[i-6,:])>0)
            ax.plot(snap_list[ind],np.log10(mstars_disk[i-6,ind]+mstars_bulge[i-6,ind])[0],'.', c='darkviolet', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkviolet', linewidth=linw1)
            prop = np.log10(mstars_disk[i-6,:])
            ind = np.where(mstars_disk[i-6,:]>0)
            ax.plot(snap_list[ind],np.log10(mstars_disk[i-6,ind])[0],'.', c='springgreen', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='springgreen', linewidth=linw1)
            prop = np.log10(mstars_bulge[i-6,:])
            ind = np.where(mstars_bulge[i-6,:]>0)
            ax.plot(snap_list[ind],np.log10(mstars_bulge[i-6,ind])[0],'.', c='khaki', linewidth=linw1)
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
            prop = np.log10(mcold[i-12,:])
            ind = np.where(mcold[i-12,:]>0)
            ax.plot(snap_list[ind],np.log10(mcold[i-12,ind])[0],'.', c='darkgreen', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkgreen', linewidth=linw1)
        elif i==15 or i==16 or i==17:
            for j in snap[i-15]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-15,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-15,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(rdisk[i-15,:])
            ind = np.where(rdisk[i-15,:]>0)
            ax.plot(snap_list[ind],np.log10(rdisk[i-15,ind])[0],'.', c='gold', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='gold', linewidth=linw1)
        elif i==18 or i==19 or i==20:
            for j in snap[i-18]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-18,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-18,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(rbulge[i-18,:])
            ind = np.where(rbulge[i-18,:]>0)
            ax.plot(snap_list[ind],np.log10(rbulge[i-18,ind])[0],'.', c='pink', linewidth=linw1)
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
        elif i==27 or i==28 or i==29:
            for j in snap[i-27]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-27,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-27,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(matom_disk[i-27,:]+matom_bulge[i-27,:])
            ind = np.where((matom_disk[i-27,:]+matom_bulge[i-27,:])>0)
            ax.plot(snap_list[ind],np.log10(matom_disk[i-27,ind]+matom_bulge[i-27,ind])[0],'.', c='purple', linewidth=linw1)
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
            prop = np.log10(mmol_disk[i-30,:]+mmol_bulge[i-30,:])
            ind = np.where((mmol_disk[i-30,:]+mmol_bulge[i-30,:])>0)
            ax.plot(snap_list[ind],np.log10(mmol_disk[i-30,ind]+mmol_bulge[i-30,ind])[0],'.', c='turquoise', linewidth=linw1)
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
        elif i==36 or i==37 or i==38:
            for j in snap[i-36]:
                ax.axvline(x=j,linestyle='--',c='red',linewidth=linw2)
            ind = np.where(mshalo[i-36,:]>0)
            if np.shape(ind)[1]!=0 and mshalo[i-36,ind[0][0]]>1e4*sim_dict['partmass']:
                ax.axvline(x=np.arange(79-np.shape(ind)[1],79,1)[0],linestyle='--',c='green',linewidth=linw2)
            prop = np.log10(vhhalo[i-36,:])
            ind = np.where(vhhalo[i-36,:]>0)
            ax.plot(snap_list[ind],np.log10(vhhalo[i-36,ind])[0],'.', c='darkred', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='darkred', linewidth=linw1)
            prop = np.log10(vchalo[i-36,:])
            ind = np.where(vchalo[i-36,:]>0)
            ax.plot(snap_list[ind],np.log10(vchalo[i-36,ind])[0],'.', c='tomato', linewidth=linw1)
            ax.plot(snap_list,prop,'-', c='tomato', linewidth=linw1)

        if i==0:
            handles1,labels = plt.gca().get_legend_handles_labels()
            patch0 = mpatches.Patch(color='darkgrey', label='Halo mass')
            patch1 = mpatches.Patch(color='black', label='Subhalo mass')
            patch2 = mpatches.Patch(color='lightblue', label='SFR burst')
            patch3 = mpatches.Patch(color='cyan', label='SFR disk')
            patch4 = mpatches.Patch(color='darkviolet', label='Stellar mass')
            patch5 = mpatches.Patch(color='springgreen', label='Stellar disk mass')
            patch6 = mpatches.Patch(color='khaki', label='Stellar bulge mass')
            patch7 = mpatches.Patch(color='orange', label='Hot gas')
            patch8 = mpatches.Patch(color='darkgreen', label='Cold gas mass')
            patch9 = mpatches.Patch(color='gold', label='Cold gas disk size')
            patch10 = mpatches.Patch(color='pink', label='Cold gas bulge size')
            patch11 = mpatches.Patch(color='teal', label='BH accretion')
            patch12 = mpatches.Patch(color='crimson', label='BH hot halo accretion')
            patch13 = mpatches.Patch(color='palegreen', label='BH sb accretion')
            patch14 = mpatches.Patch(color='olive', label='BH mass')
            patch15 = mpatches.Patch(color='purple', label='HI mass')
            patch16 = mpatches.Patch(color='lime', label='HI disk mass')
            patch17 = mpatches.Patch(color='peru', label='HI bulge mass')
            patch18 = mpatches.Patch(color='turquoise', label='H2 mass')
            patch19 = mpatches.Patch(color='orchid', label='H2 disk mass')
            patch20 = mpatches.Patch(color='salmon', label='H2 bulge mass')
            patch21 = mpatches.Patch(color='darkred', label='Halo velocity')
            patch22 = mpatches.Patch(color='tomato', label='Subhalo velocity')
#            line1 = Line2D([0],[0],label='$\Delta\mathrm{M}/\mathrm{M_{i}}<-0.9\ \mathrm{mass}\ \mathrm{drop}$',linestyle="--",color='red',linewidth=linw2)
#            line2 = Line2D([0],[0],label='$\mathrm{N}_{\mathrm{part,born}}>10^{3}$',linestyle="--",color='green',linewidth=linw2)
            line1 = Line2D([0],[0],label='Mass swapping',linestyle="--",color='red',linewidth=linw2)
            line2 = Line2D([0],[0],label='$\mathrm{N}_{\mathrm{part,born}}>10^{2}$',linestyle=":",color='green',linewidth=linw2)
            line3 = Line2D([0],[0],label='$\mathrm{N}_{\mathrm{part,born}}>10^{3}$',linestyle="--",color='green',linewidth=linw2)
            handles1.extend([patch0,patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9,patch10,patch11,patch12,patch13,patch14,patch15,patch16,patch17,patch18,patch19,patch20,patch21,patch22,line1,line2,line3])
            fig.legend(handles=handles1,loc='lower center',frameon=True,framealpha=1,bbox_to_anchor=(1.1, 0.5))

    common.savefig(outdir, fig, "MAHs-snap_wrongbranch_all.png")


def prepare_data(hdf5_data, index):
   
    #star_formation_histories and SharkSED have the same number of galaxies in the same order, and so we can safely assume that to be the case.
    #to select the same galaxies in galaxies.hdf5 we need to ask for all of those that have a stellar mass > 0, and then assume that they are in the same order.

    (h0, _, mdisk, mbulge, mhalo, mshalo, typeg, sfr_disk, sfr_burst,
     burst_mode,id_subhalo, id_halo, mbh, mhot, subhid, matom_bulge,
     matom_disk, mmol_bulge, mmol_disk, mgas_burst, mgas, rcomb, rbulge,
     rdisk, bh_accretion_hh, bh_accretion_sb, vchalo, vhhalo) = hdf5_data
    
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
    gal_props = np.zeros(shape = (ngals, 26))

    gal_props[:,0] = mdisk[ind]
    gal_props[:,1] = mbulge[ind]
    gal_props[:,2] = mhalo[ind]
    gal_props[:,3] = mshalo[ind]    
    gal_props[:,4] = typeg[ind]
    gal_props[:,5] = (sfr_disk[ind])/1e9/h0
    gal_props[:,6] = (sfr_burst[ind])/1e9/h0
    gal_props[:,7] = burst_mode[ind]
    gal_props[:,8] = id_subhalo[ind]
    gal_props[:,9] = id_halo[ind]
    gal_props[:,10] = mbh[ind]
    gal_props[:,11] = mhot[ind]
    gal_props[:,12] = subhid[ind]
    gal_props[:,13] = matom_bulge[ind]
    gal_props[:,14] = matom_disk[ind]   
    gal_props[:,15] = mmol_bulge[ind]
    gal_props[:,16] = mmol_disk[ind]   
    gal_props[:,17] = mgas_burst[ind]   
    gal_props[:,18] = mgas[ind]   
    gal_props[:,19] = rcomb[ind]   
    gal_props[:,20] = rbulge[ind]   
    gal_props[:,21] = rdisk[ind]   
    gal_props[:,22] = bh_accretion_hh[ind]/1e9/h0  
    gal_props[:,23] = bh_accretion_sb[ind]/1e9/h0   
    gal_props[:,24] = vchalo[ind]
    gal_props[:,25] = vhhalo[ind]   
  
    return (gal_props)


def main():

    simu = sys.argv[1]
    sim_dict = choose_sim(simu)
    snap_list = np.arange(1,sim_dict['snapshots'],1)

    model = sys.argv[2]
    subvols = sys.argv[3]
    zfile = sys.argv[4]
    outdir = sys.argv[5]
    galform_dir = sys.argv[6]
    
    model_dir = os.path.join(galform_dir, simu, model)
    
    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    fields = {'Output001': ('mstars_disk', 'mstars_bulge', 'mhhalo',
                            'mhalo', 'type','mstardot', 'mstardot_burst','burst_mode',
                            'index', 'ihhalo','M_SMBH', 'mhot', 'SubhaloID',
                            'mcold_atom_bulge', 'mcold_atom', 'mcold_mol_bulge',
                            'mcold_mol','mcold_burst','mcold','rcomb',
                            'rbulge', 'rdisk', 'SMBH_Mdot_hh','SMBH_Mdot_stb',
                            'vchalo','vhhalo')}
    
    snapshots = np.arange(1,sim_dict['snapshots'],1)
    
    # Create histogram

    galformID = np.array([],dtype='int32')

    mstars_disk = np.zeros(shape=(len(ID),len(snapshots)))
    mstars_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    mhalo = np.zeros(shape=(len(ID),len(snapshots)))
    mshalo = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_disk = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_burst = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_burst_diskins = np.zeros(shape=(len(ID),len(snapshots)))
    sfr_burst_mergers = np.zeros(shape=(len(ID),len(snapshots)))
    mbh = np.zeros(shape=(len(ID),len(snapshots)))
    mhot = np.zeros(shape=(len(ID),len(snapshots)))
    matom_disk = np.zeros(shape=(len(ID),len(snapshots)))
    matom_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    matom = np.zeros(shape=(len(ID),len(snapshots)))
    mmol_disk = np.zeros(shape=(len(ID),len(snapshots)))
    mmol_bulge = np.zeros(shape=(len(ID),len(snapshots)))
    mmol = np.zeros(shape=(len(ID),len(snapshots)))
    mcold = np.zeros(shape=(len(ID),len(snapshots)))
    mcold_burst = np.zeros(shape=(len(ID),len(snapshots)))
    rbulge = np.zeros(shape=(len(ID),len(snapshots)))
    rdisk = np.zeros(shape=(len(ID),len(snapshots)))
    bh_accretion_hh = np.zeros(shape=(len(ID),len(snapshots)))
    bh_accretion_sb = np.zeros(shape=(len(ID),len(snapshots)))
    vchalo = np.zeros(shape=(len(ID),len(snapshots)))
    vhhalo = np.zeros(shape=(len(ID),len(snapshots)))

    # find central galaxy
    index = 0
    hdf5_data = read_data_galform(model_dir, snapshots[-1], fields, subvols)
    (gal_props) = prepare_data(hdf5_data, index)
    h0, volh = hdf5_data[0], hdf5_data[1]
    gal_props_z0 = gal_props

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
#            print(np.shape(np.array(gal_props_z0[:,12],dtype='int64')))
#            print(np.shape(np.unique(np.array(gal_props_z0[:,12],dtype='int64'))))
#            print(np.min(np.array(gal_props_z0[:,12],dtype='int64')))
#            print(np.max(np.array(gal_props_z0[:,12],dtype='int64')))
#            print(gal_props_z0[np.where(np.array(gal_props_z0[:,12],dtype='int64')==72000069350595),4])
#            print(np.where(np.array(gal_props_z0[:,12],dtype='int64')==78000066634938))
            sID = subhaloID[ind_h]
            ind = np.where((np.array(gal_props_z0[:,12],dtype='int64') == sID) & (np.array(gal_props_z0[:,4],dtype='int32') == 0))
            if np.shape(ind)[1]>0:
                galformID = np.append(galformID,int(gal_props_z0[ind,8]))
            else:
                galformID = np.append(galformID,0)
        print("subhaloID:",sID)
    print("galformID:",galformID)

    
    for index, snapshot in enumerate(snapshots):

        hdf5_data = read_data_galform(model_dir, snapshot, fields, subvols)
        (gal_props) = prepare_data(hdf5_data, index)
        h0, volh = hdf5_data[0], hdf5_data[1]
        gal_props_z0 = gal_props

#        fname = os.path.join(model_dir, 'iz'+str(snapshot), 'ivol'+str(subvols), 'galaxies.hdf5')
#        f = h5py.File(fname,'r')
#        group = f['Output001']
#        tjm = group['Trees/jm'][()]
#        tngals = group['Trees/ngals'][()]
#        jm = np.repeat(tjm,tngals)


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
        

        for i in range(len(ID)):

            ind_h = np.where((snap==snapshot)&(finalID_raw==ID[i]))
            if np.shape(ind_h)[1]>0:
                sID = subhaloID[ind_h]
            else:
#                if i==0:
#                    sID = 10056000066816384
#                elif i==1:
#                    sID = 10045000028229526
#                elif i==2:
#                    sID = 10007000002984013
                sID = 0
            if sID !=0 or np.shape(ind_h)[1]>0:    
                print('i:',i)
                print("sID:",sID)
                ind = np.where(np.array(gal_props_z0[:,12],dtype='int64')==sID)
                print("mhalo:",gal_props_z0[ind,2])
                print("mshalo:",gal_props_z0[ind,3])
                print("type:",gal_props_z0[ind,4])
                if np.shape(ind)[1]>0:
                    mstars_disk[i,index] = gal_props_z0[ind,0]
                    mstars_bulge[i,index] = gal_props_z0[ind,1]
                    mhalo[i,index] = gal_props_z0[ind,2]
                    mshalo[i,index] = gal_props_z0[ind,3]
                    sfr_disk[i,index] = gal_props_z0[ind,5]
                    sfr_burst[i,index] = gal_props_z0[ind,6]
                    mbh[i,index] = gal_props_z0[ind,10]
                    mhot[i,index] = gal_props_z0[ind,11]
                    matom_disk[i,index] = gal_props_z0[ind,14]
                    matom_bulge[i,index] = gal_props_z0[ind,13]
                    mmol_disk[i,index] = gal_props_z0[ind,16]
                    mmol_bulge[i,index] = gal_props_z0[ind,15]
                    mcold[i,index] = gal_props_z0[ind,18]
                    rbulge[i,index] = gal_props_z0[ind,20]
                    rdisk[i,index] = gal_props_z0[ind,21]
                    bh_accretion_hh[i,index] = gal_props_z0[ind,22]
                    bh_accretion_sb[i,index] = gal_props_z0[ind,23]
                    vchalo[i,index] = gal_props_z0[ind,24]
                    vhhalo[i,index] = gal_props_z0[ind,25]

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


    plot_mah_snap(plt, outdir, h0, mshalo, mhalo, mbh, mstars_disk, mstars_bulge, mhot, mcold, rdisk, rbulge, bh_accretion_hh, bh_accretion_sb, matom_disk, matom_bulge, mmol_disk, mmol_bulge, gal_props_z0, snap_list, str(0), sfr_burst, sfr_disk, vchalo, vhhalo)
#    plot_mah_snap(plt, outdir, obsdir, h0, mshalo, mhalo, mbh, mgas_disk, mhot, rgas_disk, bh_accretion_hh, bh_accretion_sb, matom, mmol, mbh_accreted, mean_stellar_age, mlost, mreheated, mgas_bulge, rgas_bulge, gal_props_z0, LBT, str(0), sfh_a, sfh_b, sfh_c, sfh_d)
  
if __name__ == '__main__':
    main()
