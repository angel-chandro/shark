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
"""SMF plots"""

import collections
import functools
import logging
import math

import numpy as np

import common
import utilities_statistics as us

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from astropy.cosmology import FlatLambdaCDM


observation = collections.namedtuple('observation', 'label x y yerrup yerrdn err_absolute')

logger = logging.getLogger(__name__)

##################################
# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
MpcToKpc = 1e3

##################################
# Mass function initialization
mlow = 5
mupp = 14
dm = 0.125
mbins = np.arange(mlow,mupp,dm)
xmf = mbins + dm/2.0
imf   = 'cha'

mlow2 = 5
mupp2 = 14
dm2 = 0.3
mbins2 = np.arange(mlow2,mupp2,dm2)
xmf2 = mbins2 + dm2/2.0

mlow3 = 5
mupp3 = 14
dm3 = 0.2
mbins3 = np.arange(mlow3,mupp3,dm3)
xmf3 = mbins3 + dm3/2.0

ssfrlow = -6
ssfrupp = 4
dssfr = 0.2
ssfrbins = np.arange(ssfrlow,ssfrupp,dssfr)
xssfr    = ssfrbins + dssfr/2.0

sfrlow = -3
sfrupp = 1.5
dsfr = 0.2
sfrbins = np.arange(sfrlow,sfrupp,dsfr)
xsfr    = sfrbins + dsfr/2.0

size1 = 27
size2 = 9
stext = 40
fsize = 21
saxis = 27
linw1 = 6
linw2 = 2
linw3 = 30
linw4 = 35
spad = 15
spadx = 8
spady = 8
col = 'dodgerblue'

logMstars = np.linspace(8.5,11.5,20)

def sfms_fit_popesso23(t):
    # fit in the stellar mass range (10^8.5-10^11.5Msun)
    # t is the age of the Universe in Gyr
    logMstars = np.linspace(8.5,11.5,20)
    a0 = 0.20
    a1 = -0.034
    b0 = -26.134
    b1 = 4.722
    b2 = -0.1925
    return (a1*t + b1)*logMstars + b2*(logMstars)**2 + (b0 + a0*t)

def sfms_fit_schreiber15(m,z):
    # fit in the stellar mass range (10^8.5-10^11.5Msun)
    # z is redshift
    r = np.log10(1 + z)
    m = np.log10(m/1e9)
    m0 = 0.5
    a0 = 1.5
    a1 = 0.3
    m1 = 0.36
    a2 = 2.5
    return m - m0 + a0*r - a1*(np.maximum.reduce([np.zeros(np.shape(m)),m - m1 - a2*r]))**2


def plot_sfms(plt, outdir, h0, omega_m, omega_b, obsdir, mainseqsf_1):

    cosmo = FlatLambdaCDM(H0=h0*100, Om0=omega_m, Ob0=omega_b)
    LBT_end = cosmo.lookback_time(1e10).value
    LBT_0 = cosmo.lookback_time(0).value
    LBT_1 = cosmo.lookback_time(1).value
    LBT_2 = cosmo.lookback_time(2).value

    fig = plt.figure(figsize=(size1,size2))
    gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1,1,1])
    gs.update(wspace=0, hspace=0)

    xtit="$\\rm log_{10} (\\rm M_{\\star}/M_{\odot})$"
    ytit="$\\rm log_{10}(\\rm SFR/M_{\odot} yr^{-1})$"
    ytit2=r'$\mathrm{log}_{10}\ \frac{\mathrm{SFR_{\mathrm{after}}}}{\mathrm{SFR}_{\mathrm{before}}}$'
    xmin, xmax, ymin, ymax = 8, 12.5, -3.3, 3.3
    ymin2, ymax2 = -3, 3
    xleg = xmax - 0.9 * (xmax-xmin)
    yleg = ymax - 0.5 * (ymax-ymin)
    
    ax = plt.subplot(gs[0])
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1, 1))

    index = 0
    ind = np.where(mainseqsf_1[index,0,:] != 0)
    yp = mainseqsf_1[index,0,ind]
    ydn = mainseqsf_1[index,0,ind] - mainseqsf_1[index,1,ind]
    yup = mainseqsf_1[index,2,ind] + mainseqsf_1[index,0,ind]
    ax.fill_between(xmf[ind], ydn[0], yup[0], color=col, alpha = 0.25, linestyle='solid', linewidth = 3, interpolate=True)
    ax.plot(xmf[ind], yp[0],color=col,linestyle='solid', linewidth = linw1, path_effects=[pe.Stroke(linewidth=linw1+1, foreground='k'), pe.Normal()])
 
    ax.plot(logMstars,sfms_fit_popesso23(LBT_end-LBT_0),color='orange',linestyle='dashed', linewidth = 5, label="Popesso+23 MS fit")

    #GAMA data at z<0.06
    #CATAID StellarMass_bestfit StellarMass_50 StellarMass_16 StellarMass_84 SFR_bestfit SFR_50 SFR_16 SFR_84 Zgas_bestfit Zgas_50 Zgas_16 Zgas_84 DustMass_bestfit DustMass_50 DustMass_16 DustMass_84 DustLum_50 DustLum_16 DustLum_84 uberID redshift
    ms_gama, sfr_gama = common.load_observation(obsdir, 'GAMA/ProSpect_Claudia.txt', [2,6])
    ind = np.where(sfr_gama < 1e-3)
    sfr_gama[ind] = 1e-3
    #ax.hexbin(np.log10(ms_gama), np.log10(sfr_gama), gridsize=(20,20), mincnt=5) #, cmap = 'plasma') #, **contour_kwargs)
    #us.density_contour_reduced(ax, np.log10(ms_gama), np.log10(sfr_gama), 25, 25) #, **contour_kwargs)
    
    bin_it = functools.partial(us.wmedians, xbins=xmf, nmin=10)
    toplot = bin_it(x=np.log10(ms_gama), y=np.log10(sfr_gama))
    ind = np.where(toplot[0,:] != 0)
    yp = toplot[0,ind]
    yup = toplot[2,ind] + toplot[0,ind]
    ydn = toplot[0,ind] - toplot[1,ind]
    ax.fill_between(xmf[ind], ydn[0], yup[0], color='Maroon', alpha = 0.1, linestyle='solid', linewidth = 2)
    ax.plot(xmf[ind], yp[0],color='Maroon',linestyle='dashed', linewidth = 5, label="Bellstedt+20")
    
    # individual massive galaxies from Terrazas+17
    ms, sfr, upperlimflag = common.load_observation(obsdir, 'BHs/MBH_host_gals_Terrazas17.dat', [0,1,2])
    ind = np.where(ms > 11.3)
    ax.errorbar(ms[ind], sfr[ind], xerr=0.2, yerr=0.3, ls='None', mfc='None', ecolor = 'r', mec='r',marker='s',label="Terrazas+17",markersize=8,elinewidth=2)
    ind = np.where((upperlimflag == 1) & (ms > 11.3))
    for a,b in zip (ms[ind], sfr[ind]):
        ax.arrow(a, b, 0, -0.3, head_width=0.05, head_length=0.1, fc='r', ec='r')

        
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xticks(np.arange(8,13,1))
    ax.set_xticks(np.arange(8,13,0.2),minor=True)
    ax.set_xlabel(xtit,fontsize=linw3,labelpad=spadx)
    ax.set_yticks(np.arange(-3,4,1))
    ax.set_yticks(np.arange(-3,3.4,0.2),[],minor=True)
    ax.text(xleg,yleg, 'z=0' ,fontsize=stext)
    ax.tick_params(axis='both', which='major', labelsize=saxis)
    ax.set_ylabel(ytit,fontsize=linw3,labelpad=spady)
    # Legend
    plt.tight_layout()
    ax.legend(loc=2, prop={'size': fsize}, frameon=False)

    yleg = ymax - 0.25 * (ymax-ymin)

    ax = plt.subplot(gs[1])
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, '', locators=(0.1, 1, 0.1, 1))
    ax.set_yticks([])
    index = 1
    ind = np.where(mainseqsf_1[index,0,:] != 0)
    yp = mainseqsf_1[index,0,ind]
    ydn = mainseqsf_1[index,0,ind] - mainseqsf_1[index,1,ind]
    yup = mainseqsf_1[index,2,ind] + mainseqsf_1[index,0,ind]
    ax.fill_between(xmf[ind], ydn[0], yup[0], color=col, alpha = 0.25, linestyle='solid', linewidth = 3, interpolate=True)
    ax.plot(xmf[ind], yp[0],color=col,linestyle='solid', linewidth = linw1, path_effects=[pe.Stroke(linewidth=linw1+1, foreground='k'), pe.Normal()])
    ax.plot(logMstars,sfms_fit_popesso23(LBT_end-LBT_1),color='orange',linestyle='dashed', linewidth = 5)
    mstars = np.logspace(8,13,100) 
    #ax.plot(np.log10(mstars),sfms_fit_schreiber15(mstars,1),color='olive',linestyle='dashed', linewidth = 5, label="Schreiber+15 MS fit")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xticks(np.arange(8,13,1))
    ax.set_xticks(np.arange(8,13,0.2),minor=True)
    ax.set_xlabel(xtit,fontsize=linw3,labelpad=spadx)
    ax.set_yticks(np.arange(-3,4,1),[])
    ax.set_yticks(np.arange(-3,3.4,0.2),[],minor=True)
    ax.text(xleg,yleg, 'z=1' ,fontsize=stext)
    ax.tick_params(axis='both', which='major', labelsize=saxis)
    # Legend
    plt.tight_layout()
    ax.legend(loc=2, prop={'size': fsize}, frameon=False)

    yleg = ymax - 0.25 * (ymax-ymin)

    ax = plt.subplot(gs[2])
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, '', locators=(0.1, 1, 0.1, 1))
    ax.set_yticks([])
    index = 2
    ind = np.where(mainseqsf_1[index,0,:] != 0)
    yp = mainseqsf_1[index,0,ind]
    ydn = mainseqsf_1[index,0,ind] - mainseqsf_1[index,1,ind]
    yup = mainseqsf_1[index,2,ind] + mainseqsf_1[index,0,ind]
    ax.fill_between(xmf[ind], ydn[0], yup[0], color=col, alpha = 0.25, linestyle='solid', linewidth = 3, interpolate=True)
    ax.plot(xmf[ind], yp[0],color=col,linestyle='solid', linewidth = linw1, label="Shark before", path_effects=[pe.Stroke(linewidth=linw1+1, foreground='k'), pe.Normal()])
    ax.plot(logMstars,sfms_fit_popesso23(LBT_end-LBT_2),color='orange',linestyle='dashed', linewidth = 5, label="Popesso+23")
    #ax.plot(np.log10(mstars),sfms_fit_schreiber15(mstars,2),color='olive',linestyle='dashed', linewidth = 5, label="Schreiber+15")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xticks(np.arange(8,13,1))
    ax.set_xticks(np.arange(8,13,0.2),minor=True)
    ax.set_xlabel(xtit,fontsize=linw3,labelpad=spadx)
    ax.set_yticks(np.arange(-3,4,1),[])
    ax.set_yticks(np.arange(-3,3.4,0.2),[],minor=True)
    ax.text(xleg,yleg, 'z=2' ,fontsize=stext)
    ax.tick_params(axis='both', which='major', labelsize=saxis)
    # Legend
    plt.tight_layout()

    common.savefig(outdir, fig, 'SFR_Mstars_plot_panel.pdf')


def prepare_data(hdf5_data, index, mainseqsf):

    (h0, volh, sfr_disk, sfr_burst, mdisk, mbulge) = hdf5_data

    bin_it_2sigma = functools.partial(us.wmedians_2sigma, xbins=xmf)
    
    mass          = np.zeros(shape = len(mdisk))
    ind = np.where((mdisk+mbulge) > 0.0)
    mass[ind] = np.log10(mdisk[ind] + mbulge[ind]) - np.log10(float(h0))
    logger.debug('number of galaxies with mstars>0 and max mass: %d, %d', len(mass[ind]), max(mass[ind]))
    
    ind = np.where((sfr_disk+sfr_burst > 0) & (mdisk+mbulge > 0) & ((sfr_disk+sfr_burst)/(mdisk+mbulge) > 0))
    mainseqsf[index,:] = bin_it_2sigma(x=mass[ind], y=np.log10((sfr_disk[ind]+sfr_burst[ind])/h0/GyrToYr))

    return mass

def main(modeldir, outdir, redshift_table, subvols, obsdir):

    model = modeldir.split('/')[-1]
    sim = modeldir.split('/')[-2]

    zlist = (0, 0.5, 1, 2, 3, 4)

    plt = common.load_matplotlib()

    mainseqsf     = np.zeros(shape = (len(zlist), 3, len(xmf)))

    fields = {'galaxies': ('sfr_disk', 'sfr_burst', 'mstars_disk', 'mstars_bulge')}
    fields_c = {'cosmology': ('omega_m', 'omega_b')}

    for index, snapshot in enumerate(redshift_table[zlist]):
        hdf5_data = common.read_data(modeldir, snapshot, fields, subvols)
        mass = prepare_data(hdf5_data, index, mainseqsf)

        subvol = set()
        subvol.add(0)
        hdf5_data = common.read_data(modeldir, redshift_table[zlist[0]], fields_c, subvol)
        (h0, volh, omega_m, omega_b) = hdf5_data

    plot_sfms(plt, outdir, h0, omega_m, omega_b, obsdir, mainseqsf)
        


if __name__ == '__main__':
    main(*common.parse_args())
