##############################################
######## Computes kSZ power spectrum #########
### Copyright Stephane Ilic & Adélie Gorce ###
##############################################

import camb
from camb import model, initialpower
import numpy as np
import sys
import multiprocessing
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 

from parameters import *
from functions import *

if debug:
    print('\nYou are running in debugging mode')
if not late_time:
    print('\nYou are running for patchy kSZ only')

##############################################
############## CAMB ##########################
##############################################

print("\nRunning CAMB...")

z3=np.linspace(0,z_max,100)         
x3 = xe(z3,zend,zre)
tau = xe2tau(z3,x3)[-1]
print("tau = %.5f" %(tau))

pars = camb.CAMBparams()
pars.set_cosmology(H0=h*100,ombh2 = obh2,omch2 = och2,TCMB=T_cmb,tau=tau)
pars.InitPower.set_params(ns=n_s,r=0,As=A_s)
pars.WantTransfer = True
pars.Reion.set_tau(tau)
pars.set_for_lmax(ell_max_CMB, lens_potential_accuracy=0);
pars.set_dark_energy()
data= camb.get_background(pars)
results = camb.get_results(pars)

##############################################
#### Cosmo functions & derived parameters ####
##############################################

print("Extract info from CAMB run...")
print('Computing TT TE EE Cl')
results.calc_power_spectra(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
ls = np.arange(totCL.shape[0])
### Hubble function (=adot/a) in SI units [s-1] (CAMB gives km/s/Mpc)
H = np.vectorize(lambda z: results.hubble_parameter(z)/Mpckm)
### Growth rate f
f = np.vectorize(lambda z: data.get_redshift_evolution(0.1,z,['growth']).flatten())
### Comoving distance / conformal time in Mpc
D_C = np.vectorize(lambda z: results.comoving_radial_distance(z))
## Linear matter power spectrum P(z,k) - no hubble units, result in (Mpc)^3
kmax_pk = ell_max_kSZ/D_C(z_min)
assert (kmax_pk < 2e3),'Error, k too large for P(k) extrapolation, need to modify either ell_max or z_min'
print('Interpolating linear power spectrum')
interp_l=camb.get_matter_power_interpolator(pars,nonlinear=False,kmax=kmax_pk,hubble_units=False,k_hunit=False,zmax=z_max, var1=model.Transfer_nonu,var2=model.Transfer_nonu)
Pk_lin=np.vectorize(lambda k,z : interp_l.P(z,k))
### Non-linear matter power spectrum
print('Interpolating non linear power spectrum')
interp_nl=camb.get_matter_power_interpolator(pars,nonlinear=True,kmax=kmax_pk,hubble_units=False,k_hunit=False,zmax=z_max, var1=model.Transfer_nonu,var2=model.Transfer_nonu)
Pk = np.vectorize(lambda k, z : interp_nl.P(z,k))
## Hydrogen number density function in SI units [m-3]
n_H = lambda z: nh*(1.+z)**3.
print('Finished computing cosmo functions')

################################
#### Arrays for integration ####
################################
print("\nPrepare for kSZ calculation...")

# free electrons power spectrum
def Pee(k,z):
    return (fH-xe(z))*W(k,xe(z)) + xe(z)*bdH(k,z)*Pk(k,z)

# computations below take 6.5 secs
b_del_e_integ = np.sqrt(Pee(kp_integ[:, None], z_integ[:, None, None])/Pk(kp_integ[:, None], z_integ[:, None, None])) #electrons bias
eta_z_integ = D_C(z_integ)  # comoving distance to z in [Mpc]
detadz_z_integ = c / H(z_integ)  # Hubble parameter in SI units [m]
f_z_integ = f(z_integ)  # growth rate, no units
adot_z_integ = (1. / (1. + z_integ)) * H(z_integ)  # in SI units [s-1]
n_H_z_integ = n_H(z_integ)  # number density of baryons in SI units [m-3]
x_i_z_integ = xe(z_integ) # reionisation history
tau_z_integ = xe2tau(z_integ,x_i_z_integ)  # thomson optical depth
Pk_lin_integ = Pk_lin(kp_integ[:, None],z_integ[:, None, None]) #linear matter power spectrum

if debug:

    import matplotlib.pyplot as plt
    import matplotlib as m
    m.rcParams.update({'font.size': 15})
    
    k2=np.logspace(min_logkp,max_logkp,100)
    z_range=np.arange(0.,z_max+1.,step=1)
    k_range = [0.05,0.1,0.2,0.5,1.,2.,5.,10.,20.]

    plt.figure(figsize=(9,8))
    plt.plot(z3,xe(z3))
    plt.xlabel(r'Redshift $z$')
    plt.ylabel(r'Ionisation level $x_e(z)$')
    plt.xlim(z_min,z_max)
    plt.tight_layout()
    plt.savefig(folder+'/x_e_z.png')

    cmap = m.cm.get_cmap('coolwarm')
    norm = m.colors.LogNorm(vmin=5e-2, vmax=1)
    plt.figure(figsize=(14,8))
    plt.axhline(1,color='k',lw=.8,ls='-')
    for u,z in enumerate(z_range):
        plt.loglog(k2,Pee(k2,z),lw=1.5,color=cmap(norm(xe(z))))
    plt.ylim(1e-6,1e6)
    plt.xlim(k2.min(),k2.max())
    plt.xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
    plt.ylabel(r'$P_{ee}(k,z) [\mathrm{Mpc}^3]$')
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    c_bar=plt.colorbar(sm,fraction=0.05, norm=norm)
    c_bar.set_label(r'$x_e$')
    plt.tight_layout()
    plt.savefig(folder+'/Pee_vs_k_kSZ.png')

    cmapz = m.cm.get_cmap('PuRd')
    normz = m.colors.LogNorm(vmin=np.min(k_range)/10, vmax=np.max(k_range))
    plt.figure(figsize=(10,8))
    plt.axhline(1,lw=.8,color='k')
    plt.axvline(0.5,color='k',lw=1,ls=':')
    for u,k in enumerate(k_range):
        plt.loglog(xe(z3),Pee(k,z3),lw=1.,color=cmapz(normz(k)))
    plt.xlabel(r'Ionisation level $x_e(z)$')
    plt.ylabel(r'$P_{ee}(k,z) [\mathrm{Mpc}^3]$')
    plt.xlim(xe_recomb,fH)
    plt.ylim(1e-3,1e5)
    smz = plt.cm.ScalarMappable(cmap=cmapz,norm=normz)
    smz.set_array([])
    c_bar=plt.colorbar(smz,fraction=0.05, norm=normz)
    c_bar.set_label(r'$k\, [\mathrm{Mpc}^{-1}]$')
    plt.tight_layout()
    plt.savefig(folder+'/Pee_vs_x_kSZ.png')


########################
#### C_ell fonction ####
########################
def C_ell_kSZ(ell,late=late_time,debug=debug):
    ### Preliminaries
    # in [Mpc-1]
    k_z_integ = ell / eta_z_integ

    # in [Mpc-1]
    k_min_kp = np.sqrt(k_z_integ[:, None, None]**2 + kp_integ[:, None]**2. - 2. * k_z_integ[:, None, None] * kp_integ[:, None] * mu)

    ### Compute I_tot1 and I_tot2, in [Mpc^2]
    Pee_min_kp = Pee(k_min_kp,z_integ[:, None, None])
    I_e = ( Pee_min_kp / kp_integ[:, None]**2.) - (np.sqrt(Pee_min_kp/Pk(k_min_kp,z_integ[:, None, None])) *  b_del_e_integ * Pk_lin(k_min_kp,z_integ[:, None, None]) / k_min_kp**2)
    
    ### Compute Delta_B^2 integrand, in [s-2.Mpc^2]
    Delta_B2_integrand = (
        k_z_integ[:, None, None]**3. / 2. / np.pi**2. *
        (f_z_integ[:, None, None] * adot_z_integ[:, None, None])**2. *
        kp_integ[:, None]**3. * np.log(10.) * np.sin(th_integ) / (2. * np.pi)**2. *
        Pk_lin_integ * (1. - mu**2.) * I_e
    )
    ### Compute Delta_B^2, in [s-2.Mpc^2]
    Delta_B2 = simps(simps(Delta_B2_integrand, th_integ), np.log10(kp_integ))

    ### Compute C_kSZ(ell) integrand, unit 1
    C_ell_kSZ_integrand = (
        8. * np.pi**2. / (2. * ell + 1.)**3. * (s_T / c)**2. *
        (n_H_z_integ * x_i_z_integ / (1. + z_integ))**2. *
        Delta_B2 *
        np.exp(-2. * tau_z_integ) * eta_z_integ * detadz_z_integ * Mpcm**3.
    )

    ### Compute C_kSZ(ell), no units
    result = trapz(C_ell_kSZ_integrand,z_integ)
    if late:
        g = z_integ >= zend 
        result_p = trapz(C_ell_kSZ_integrand[g],z_integ[g])    
        res = [result_p,result]
    else:
        res = [result]      

    if debug:
        sys.stdout.write("...C(ell=%i) = " %ell)
        for r in res:
            sys.stdout.write(' %.2e' %r)
        print(' ')

    del(Delta_B2_integrand)
    del(I_e)

    return res

#######################################
### Parallel computation of C_ells ####
#######################################
print('\nComputing for %i l on range [%i,%i] with %i threads' %(n_ells_kSZ,ell_min_kSZ,ell_max_kSZ,n_threads))
print("Begin kSZ calculation...")
ells = np.linspace(ell_min_kSZ, ell_max_kSZ, n_ells_kSZ, dtype=int)
if debug:
    ells = np.random.randint(ell_min_kSZ, ell_max_kSZ, n_ells_kSZ)
C_ells_kSZ = np.array(multiprocessing.Pool(n_threads).map(C_ell_kSZ, ells))

######################
### Write outputs ####
######################
### Writing kSZ C_ells
np.savetxt(outroot+"_kSZ_Cells.txt",
    np.c_[ells,C_ells_kSZ],
    header="ell   kSZ tot   kSZ patchy    NB: dimensionless C_l's",delimiter=' ',fmt='%.5e')
### Writing TT/TE/EE unlensed C_ells
np.savetxt(outroot+"_CMB_Cells.txt",
    np.transpose([ls, unlensedCL[:,0],unlensedCL[:,1],unlensedCL[:,3]]),
    header="ell   TT   EE   TE    NB: dimensionless [l(l+1)/2pi] C_l's")
### Writing TT/TE/EE lensed C_ells
np.savetxt(outroot+"_lensed_CMB_Cells.txt",
    np.transpose([ls, totCL[:,0],totCL[:,1],totCL[:,3]]),
    header="ell   TT   EE   TE    NB: dimensionless [l(l+1)/2pi] C_l's")

