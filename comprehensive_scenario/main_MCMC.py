import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, integrate
import sys, emcee
from multiprocessing import Pool
from highz import *
import highz_config as hz
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

data=str(sys.argv[1])
nthreads = int(sys.argv[2])

def xerob2tauR(z,xe): #xe:ionization rate of the Universe ; z = list in descending order (from 30 to 0 for instance)
    """
    computes tau(z) integrated from xe(z) 
    """
    dz = z[1]-z[0]

    csurH = 0.925e28 / hz.h  #cm
    Hdezsurc = (hz.Om*(1+z)**3.+hz.Ol)**(0.5) /csurH

    eta=1 #eta=1 for now
    integ2 = (1+z)**2/Hdezsurc * xe *hz.sigt* hz.nh * (1+eta*hz.Yp/4/hz.Xp)
    eta=2 #for z<4 because of the double ionization of He
    w = z<4 
    integ2[w] = (1+z[w])**2/Hdezsurc[w] * xe[w] *hz.sigt * hz.nh * (1+eta*hz.Yp/4/hz.Xp)
    taudez2 = np.cumsum(integ2)*dz

    return taudez2[-1]


#likelihood
def lnlikeR(theta, x, y, yerr,xx,yx,yxerr,*args):

    #intial parameters for the fit (result parameters from the paper)
    a,b,c,d = theta

    chi  = -1e10
    if (nion(x,theta,*args)==0).any():
        #avoid calculating log of negative values
        return -1e5


    #rho model
    model = np.log10(sfrd(x,theta,*args))
    chi = -0.5*(np.sum((y-model)**2/yerr**2 ))

    #calculation of QHII from rho model
    model2    = np.zeros(xx.size)
    y0        = 0.0001
    xxx       = np.sort(xx)[::-1]
    ind       = np.argsort(xx)[::-1]
    z         = np.linspace(30, 0, 301)
    xe1       = np.zeros(z.size)
    xe1        = QHII(z,y0,(a,b,c,d),*args)
    interpfunc = interpolate.interp1d(z,xe1, kind='linear')
    model2     = interpfunc(xxx)
    chi2 = -0.5*(np.sum((yx[ind]-model2)**2/yxerr[ind]**2 ))

    #tau
    tau = xerob2tauR(z,xe1) #asymptotic value of tau
    chi3 = -0.5*((hz.tau-tau)**2/hz.sigtau**2 ) 

    if (data=='100'):
        return chi
    elif (data=='010'): #only nion data is considered
        return chi2
    elif (data=='110'):#rho and QHII data considered
        return chi+chi2
    elif (data=='101'):#rho data and tau contraint considered
        return chi+chi3
    elif (data=='011'):#QHII data and tau constraint considered
        return chi2+chi3
    elif (data=='001'):#tau constraint only
        return chi3
    elif (data=='111'):
        return chi+chi2+chi3
    else:
        raise Error("You have to choose a model")

def lnpriorR(theta):
    a,b,c,d=theta
    if 0.1 < a < 2. and 0.001 < b < 15. and 0.001 < c < 10. and 0.001 < d < 15. and b - d < -0.001:
        return 0.0
    return -np.inf

def lnprobR(theta, x, y, yerr,xx,yx,yxerr, *args):
    lp = lnpriorR(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikeR(theta,x,y,yerr,xx,yx,yxerr,*args)

def loading_verbose(msg):
    sys.stdout.write('\r'+msg)
    sys.stdout.flush()

#################################################### MAIN

########################################## INI
nsteps     = 50000 #number of iterations of the chain
nparchains = 20 #number of chains running at the same time
# nthreads   = 1 #number of processors
filechain = './Outfiles/chains_'+str(data)+'.dat'

########################################## DATA
zdi,rhodim,rhodi, rhodip=np.loadtxt("./Data/data.ir.txt",skiprows=1,unpack=True)
zdu,rhodum,rhodu, rhodup=np.loadtxt("./Data/data.uv.txt",skiprows=1,unpack=True)

zdd   = np.hstack((zdi,zdu))
rhodd = np.hstack((rhodi,rhodu))
errdd = np.hstack((rhodip,rhodup))-np.hstack((rhodi,rhodu))

fzrob, fxerob, fdxerobp, fdxerobm = np.loadtxt("./Data/constraints_readble.txt",usecols=(0,1,2,3),dtype=float,unpack=True)

########################################## START

y0     = 0.0001# initial condition
z      = np.linspace(30, 0.1, 301)
xe1, xe2, xe3    = np.zeros(301),np.zeros(301),np.zeros(301)

########################################## MCMC

ndim, nwalkers = 4,nparchains

#initial position
pos     = [hz.theta_init + 1e-2*np.random.randn(ndim)*[0.1,10.,10.,10.] for i in range(nwalkers)]
with Pool(nthreads) as pool: 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobR, args=(zdd, rhodd,errdd,fzrob,fxerob,np.max([fdxerobp, fdxerobm],0)),pool=pool)
    sampler.run_mcmc(pos, nsteps,progress=True)

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
samples = sampler.get_chain(flat=True) 
like = sampler.get_log_prob(flat=True)

# plot walkers
labels = [r"$a$", r"$b$", r"$c$", r"$d$"]
fig, axes = plt.subplots(ndim, figsize=(ndim*4, ndim*2), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel("Step number")
plt.tight_layout()

tau = sampler.get_autocorr_time()
if np.isnan(tau).any():
    tau=1000
flat_samples = sampler.get_chain(thin=2, flat=True, discard=int(np.max(np.round(tau)*2)))
flat_like = sampler.get_log_prob(thin=2, flat=True, discard=int(np.max(np.round(tau)*2)))
tau=np.zeros(flat_samples.shape[0])
np.savetxt(filechain,np.c_[flat_samples,tau,flat_like], header='a b c d tau like, %i iterations' %(flat_like.size))

print('ML parameters:')
print('a = %.4f +/- %.4f' %(np.median(flat_samples[:,0]),np.std(flat_samples[:,0])))
print('b = %.2f +/- %.2f' %(np.median(flat_samples[:,1]),np.std(flat_samples[:,1])))
print('c = %.2f +/- %.2f' %(np.median(flat_samples[:,2]),np.std(flat_samples[:,2])))
print('d = %.2f +/- %.2f' %(np.median(flat_samples[:,3]),np.std(flat_samples[:,3])))

################### Computing tau
print('Computing tau...')
def xe_tau(a1,b1,c1,d1):
    xe = QHII(z,y0,tuple((a1,b1,c1,d1)))
    csurH = 0.925e28 / hz.h  #cm
    Hdezsurc = (hz.Om*(1+z)**3.+hz.Ol)**(0.5) /csurH

    eta=1 #eta=1 for now
    integ2 = (1+z)**2/Hdezsurc * xe *hz.sigt* hz.nh * (1+eta*hz.Yp/4/hz.Xp)
    eta=2 #for z<4 because of the double ionization of He
    w = z<4 
    integ2[w] = (1+z[w])**2/Hdezsurc[w] * xe[w] *hz.sigt * hz.nh * (1+eta*hz.Yp/4/hz.Xp)

    return integ2

with Pool(nthreads) as pool:
    xe3 = pool.starmap(xe_tau, [(a1,b1,c1,d1) for a1,b1,c1,d1 in flat_samples])
dz = z[1]-z[0]
tau=-(np.cumsum(xe3,axis=1)*dz)[:,-1]
print('Done.')

################### Writing out
print("Writing out")   
np.savetxt(filechain,np.c_[flat_samples,tau,flat_like], header='a b c d tau like, %i iterations' %(flat_like.size))

#############################################################3
