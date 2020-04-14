import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy import integrate
import sys
from highz import *
import highz_config as hz
import triangleme2 as triangleme
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# plt.ion()
colors = ['#1f77b4','#d62728',  '#ff7f0e', '#2ca02c','#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
cmaps = ['Blues','Reds','Oranges','Greens','Purples','copper']
#############################Robertson

def loading_verbose(msg):
    sys.stdout.write('\r'+msg)
    sys.stdout.flush()

def readmcmc(data,subn=100000,extent=[]):

    ns = len(data)
    print('\nTriangle plot...')

    labels = [r"$a$", r"$b$", r"$c$", r"$d$",r"$\tau$"]
    if (0<len(extent)<5):
        raise Error('Wrong format for extents')

    fig00 = plt.figure(figsize=(10,10))
    # plot_name = './Figures/triangle_plot'
    ################### READING MCMC
    for u,data1 in enumerate(data):
        print('\nImporting data %s' %(data1))

        outfile = './Outfiles/chains_'+str(data1)+'.dat' 
        samples      = np.loadtxt(outfile,unpack=True,usecols=(0,1,2,3,4)) #includes tau
        # print(str(data1),np.median(samples,axis=1))

        print('a = %.4f +/- %.4f' %(np.median(samples[0,:]),np.std(samples[0,:])))
        print('b = %.2f +/- %.2f' %(np.median(samples[1,:]),np.std(samples[1,:])))
        print('c = %.2f +/- %.2f' %(np.median(samples[2,:]),np.std(samples[2,:])))
        print('d = %.2f +/- %.2f' %(np.median(samples[3,:]),np.std(samples[3,:])))
        print('tau = %.4f +/- %.4f' %(np.median(samples[4,:]),np.std(samples[4,:])))

        subn = min(subn,samples.shape[1])
        subsamples = samples[:,np.random.randint(0,samples.shape[1], size=subn)] 
        if (len(extent)==0):
            triangleme.corner(subsamples.T, labels=labels, plot_contours=True, plot_datapoints=False, plot_ellipse=False, ls='.',cmap=cmaps[u],color=colors[u],lw=.8,fig=fig00)
        else:
            triangleme.corner(subsamples.T, labels=labels, plot_contours=True, plot_datapoints=False, plot_ellipse=False, ls='.', extents=extent,cmap=cmaps[u],color=colors[u],lw=.8,fig=fig00)
        plot_name = plot_name+'_'+str(data1)

    # plt.savefig(plot_name+'.png') 

    return

def model_plots(data,nrand=10000,CL=95):

    ns = len(data)
    percentile1=(100-CL)/2
    percentile2=CL+(100-CL)/2
    from xetanh import xe_tanh,xe_asym
    # matplotlib.rcParams.update({'font.size': 18})
    print('\nPlotting x_e, rho and tau...')

    ########################################## DATA
    zdi,rhodim,rhodi, rhodip=np.loadtxt("./Data/data.ir.txt",skiprows=1,unpack=True)
    zdu,rhodum,rhodu, rhodup=np.loadtxt("./Data/data.uv.txt",skiprows=1,unpack=True)
    rhodd = np.hstack((rhodi,rhodu))
    errdd = np.hstack((rhodip,rhodup))-np.hstack((rhodi,rhodu))

    z_data, xe_data, dxep, dxem = np.loadtxt("./Data/constraints_readble.txt",usecols=(0,1,2,3),dtype=float,unpack=True)
    ref = np.loadtxt("./Data/constraints_readble.txt",usecols=(4),dtype=str,unpack=True)

    z      = np.linspace(0,30, 300)
    y0     = 0.0001# initial condition
    rec = [0.1, 0.1, 0.85, 0.85]

    # xe figure
    figx = plt.figure(figsize=(12,8))
    axx = figx.add_axes(rec)
    axx.set_xlim(4,16)
    axx.set_xlabel(r'Redshift $z$',fontsize=20)
    axx.set_ylim(0.,hz.fH)
    axx.set_ylabel(r'IGM ionised fraction $x_e$',fontsize=20)
    #data from LAEs
    w=np.where((ref=='LAE') & (dxep==0))
    uplims = np.zeros(z_data.shape)
    uplims[w] = True
    axx.errorbar(z_data[w], xe_data[w], yerr=[dxem[w],dxep[w]], fmt='*',uplims=uplims[w],ecolor='k',color='k',elinewidth=1.,ms=8,capsize=3,alpha=.7)
    w=np.where((ref=='LAE') & (dxep!=0))
    axx.errorbar(z_data[w], xe_data[w], yerr=[dxem[w],dxep[w]], fmt='*',ecolor='k',color='k',elinewidth=1.,ms=8, label=r'Ly-$\alpha$ emitters',capsize=3,alpha=.7)
    #Data from dumping wings
    w=np.where((ref=='QSOs') & (dxep==0))
    uplims = np.zeros(z_data.shape)
    uplims[w] = True
    axx.errorbar(z_data[w], xe_data[w], yerr=[dxem[w],dxep[w]], fmt='s',uplims=uplims[w],ecolor='k',color='k',elinewidth=1.,ms=5,capsize=3,alpha=.7)
    w=np.where(ref=='QSOs')
    axx.errorbar(z_data[w], xe_data[w], yerr=[dxem[w],dxep[w]], fmt='s',ecolor='k',color='k',elinewidth=1.,ms=5, label='QSO spectra',capsize=3,alpha=.7)
    w=np.where(ref=='GRB')
    axx.errorbar(z_data[w], xe_data[w], yerr=[dxem[w],dxep[w]], fmt='D',ecolor='k',color='k',elinewidth=1.,ms=5,label='GRB afterglows',capsize=3,alpha=.7)
    #tanh model for xe
    axx.plot(z,xe_tanh(z,8.5,0.5,helium=hz.HeliumI,helium2=hz.HeliumII),'purple',linestyle='--',linewidth=1.5,label=r'Symmetric model from P16')
    #asymmetric model for xe
    axx.plot(z,xe_asym(z,6.1,6.6,helium=hz.HeliumI,helium2=hz.HeliumII),color='purple',linestyle='-.',linewidth=1.5,label=r'Asymmetric model from P16')
    
    # rho figure
    figr = plt.figure(figsize=(12,8))
    axr = figr.add_axes(rec)
    axr.set_xlim(0.1,15)
    axr.set_xlabel(r'Redshift $z$',fontsize=20)
    axr.set_ylabel(r'$\mathrm{log}(\rho_{\mathrm{SFR}})\ [\mathrm{M}_{\odot} \mathrm{yr}^{-1} \mathrm{Mpc}^{-3}]$',fontsize=20)
    axr.set_ylim(-4,0)
    #plot of the UV and IR data
    axr.errorbar(zdi, rhodi, yerr=[rhodi-rhodim,rhodip-rhodi], fmt='o',elinewidth=1,capsize=3, color='purple', ecolor='purple', label='IR luminosity density',alpha=.7)
    axr.errorbar(zdu, rhodu, yerr=[rhodu-rhodum,rhodup-rhodu], fmt='o',elinewidth=1,capsize=3, color='plum', ecolor='plum', label='UV luminosity density',alpha=.7)

    # tau figure
    figt = plt.figure(figsize=(8,8))
    axt = figt.add_axes([0.12, 0.1, 0.84, 0.85])
    axt.set_xlim(0.0,0.10)
    axt.set_ylim(0.,1.1)
    axt.set_xlabel(r'Thomson optical depth $\tau$',fontsize=24)
    axt.set_ylabel(r'Probability distribution',fontsize=24)
    # axt.set_ylim(0.,1.5)
    #gaussian with Planck value
    x0=np.arange(1000)/10000.
    gg= np.exp(-(x0-hz.tau)**2/2./hz.sigtau**2)/np.sqrt(2.*np.pi)
    gg=gg/np.max(gg)
    axt.plot(x0, gg,color='purple',lw=1.5,label='Planck+2018')

    ################### READING MCMC
    # endstr = '.png'
    for ii,data1 in enumerate(data):

        print('\nImporting data %s' %(data1))
        outfile = './Outfiles/chains_'+str(data1)+'.dat' 
        a,b,c,d,tau,like      = np.loadtxt(outfile,unpack=True)
        samples  = np.c_[a,b,c,d]
        # endstr = '_'+str(data1)+endstr

        rho, xe = np.zeros((z.size,nrand)), np.zeros((z.size,nrand))
        u=0
        for a,b,c,d in samples[np.random.randint(len(samples), size=nrand)]:#[np.random.randint(len(samples), size=nrand)]:
            theta=(a,b,c,d)
            rho[:,u] = sfrd(z[::-1],(a,b,c,d))
            xe1 = QHII(z[::-1],y0,theta).flatten()
            xe1[xe1>hz.fH] = hz.fH
            xe[:,u]= xe1
            msg = str('Computing confidence intervals... %i%% done' %((u+1)/nrand*100))
            loading_verbose(msg)
            u=u+1
        msg = str('Computing confidence intervals... 100% done.')
        loading_verbose(msg)

        axr.fill_between(z[::-1], np.log10(np.percentile(rho,percentile2,axis=1)), np.log10(np.percentile(rho,percentile1,axis=1)), color=colors[ii],alpha=.3)
        axr.plot(z[::-1],np.log10(np.median(rho,axis=1)),color=colors[ii],lw=2)#,label=str(data1))

        axx.fill_between(z[::-1], np.percentile(xe,percentile2,axis=1), np.percentile(xe,percentile1,axis=1), color=colors[ii],alpha=.3)
        axx.plot(z[::-1],np.median(xe,axis=1),color=colors[ii],lw=2)#,label=str(data1))

        values,bins= np.histogram(tau,bins=100,density=True)#,label=str(data1))
        bins=(bins[1:]+bins[:-1])/2
        axt.plot(bins,values/np.max(values),drawstyle='steps-mid',color=colors[ii],lw=1.5)
        # axt.axvline(np.median(tau),color=colors[ii],lw=1.5)

    axr.legend(loc=1, frameon=False,fontsize=18)
    # figr.savefig('./Figures/rho_SFR'+endstr)

    axx.legend(loc='best', frameon=False,fontsize=18)
    # figx.savefig('./Figures/xe'+endstr)

    axt.legend(loc='best', frameon=False,fontsize=18)
    # figt.savefig('./Figures/tau_distri'+endstr)

    print(' ')

    return
