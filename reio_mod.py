# Reionisation module
# MD 05/2015 from idl routines from SI and MD

from scipy.integrate import cumtrapz
import numpy as np
from astropy import cosmology, constants, units

#### PARAMETERs

##########################
#### Cosmo parameters ####
##########################
h = 0.6774000
Om_0 = 0.309
Ob_0 = 0.049
T_cmb = 2.7255
cos = cosmology.FlatLambdaCDM(H0=h*100, Tcmb0=T_cmb, Ob0=Ob_0, Om0=Om_0)
Yp = 0.24524332588411976  # 0.2453
Xp = 1-Yp
mh = constants.m_n.value #kg
rhoc = cos.critical_density0.si.value #kg m-3
nh = Xp*Ob_0*rhoc/mh  # m-3
xe_recomb= 2.0943995348454196E-004 #1.0e-4



not4                            = 3.9715 #eta

###################
#### Constants ####
###################
s_T = constants.sigma_T.value    # sigma_thomson in SI units [m^2]
c = constants.c.value   # speed of light in SI units [m.s-1]
Mpcm = (1.0 * units.Mpc).to(units.m).value  # one Mpc in [m]
Mpckm = Mpcm / 1e3


#########################################################
# xe_asym (z,zend,alpha,z_early,helium)                 #
#########################################################

def xe_asym(z,zend=5.5,zre=7.0,z_early=20.,helium1=True,helium2=True,
            helium2_redshift= 3.5, helium2_deltaredshift=0.5, helium2_redshiftstart=5.0,
            xe_recomb=1.7e-4):
    """
    Computes the redshift-asymmetric parameterisation of xe(z) in Douspis+2015
    Parameters:
        zend : redshift at wich reionisation ends
        zre : midpoint (when xe = 0.50)
        z_early : redshift aroiund which the first sources form (taken to 20)
        helium, helium2 : to include helium first and second reionisation or not
        xe_recomb : ionised fraction leftover after recombination (default is 1.7e-4)
        CAREFUL z must not go further than z_early

    """
    fH = 1.
    if helium1:
        fHe = Yp/(3.9715*(1-Yp))
        fH+=fHe
    alpha = np.log(1./2./fH)/ np.log((z_early-zre)/(z_early-zend))
    frac=0.5*(np.sign(zend-z)+1) + 0.5*(np.sign(z-zend)+1)*abs((z_early-z)/(z_early-zend))**(alpha)
    # frac=0.5*(np.sign(zend-z)+1) + 0.5*(np.sign(z-zend)+1)*abs((z_early-z)/(z_early-zend))**(alpha)

    xe = (fH-xe_recomb)*frac +xe_recomb
    if helium2:
        assert helium1, "Need to set both He reionisation to True, cannot have HeII without HeI"
        a = np.divide(1,z+1.)
        deltayHe2 = 1.5*np.sqrt(1+helium2_redshift)*helium2_deltaredshift
        VarMid2    = (1.+helium2_redshift)**1.5
        xod2 = (VarMid2 - 1./a**1.5)/deltayHe2
        tgh2 = np.tanh(xod2) # check if xod<100
        xe += (fHe-xe_recomb) * (tgh2+1.)/2.
    return xe


def xe_tanh(z,ze=7.0,deltaz=0.5, helium1=True, helium1_redshift=7,
                        helium1_deltaredshift=.5,
                        helium2=True, helium2_redshift=3.5,
                        helium2_deltaredshift=.7, xe_recomb=xe_recomb):

    """
    computes the redshift-symmetric parameterisation of xe(z) (tanh)
    as function of z_reio (midpoint) and delta_z (duration)
    """

    fH = 1.
    if helium1:
        fHe = Yp/(3.9715*(1-Yp))
        #fH+=fHe
        # print(fHe)

    deltay    = 1.5*np.sqrt(1+ze)*deltaz
    VarMid    = (1.+ze)**1.5

    xod = ((1+ze)**1.5 - (1+z)**1.5)/deltay
    tgh = np.tanh(xod)

    xe = (fH-xe_recomb)*(tgh+1.)/2.+xe_recomb

    if (helium1):
        a = 1./(z+1.)
        
        deltayHe1 = 1.5*np.sqrt(1+helium1_redshift)*helium1_deltaredshift
        VarMid1    = (1.+helium1_redshift)**1.5

        xod1 = (VarMid1 - (1./a**1.5))/deltayHe1
        tgh1 = np.tanh(xod1)
        #print('redshift:', z, 'Adelies fHeI:', fHe, 'Adelies tgh1:', (tgh1+1.)/2.)
        xe += fHe*(tgh1+1.)/2.

    if (helium2):
        a = 1./(z+1.)
        deltayHe2 = 1.5*np.sqrt(1+helium2_redshift)*helium2_deltaredshift
        VarMid2    = (1.+helium2_redshift)**1.5
        xod2 = (VarMid2 - 1./a**1.5)/deltayHe2
        tgh2 = np.tanh(xod2)

        xe += fHe*(tgh2+1.)/2.


    return xe

#########################################################
#   xe_to_tau(z,xe,cosmo=cosmo)                         #
#########################################################

def xe2tau(z,xe):
    """
    computes tau(z) integrated from xe(z)
    Params:
        xe : ionization rate of the Universe
        z : list in descending order (from 30 to 0 for instance)
    """
    z = np.sort(z)
    xe = np.sort(xe)[::-1]
    csurH = c*100/cos.H(0).si.value  #cm
    Hdezsurc = cos.H(z).si.value/(c*100) #cm-1

    eta=1 #eta=1 for now
    integ2 = c * s_T * nh * xe / cos.H(z).si.value * (1+z)**2 * (1+eta*Yp/4/Xp)
    taudez2 = cumtrapz(integ2[::-1], z, initial=0)[::-1]

    return taudez2

def xe2tau_bis(z, xe):
    """
    computes tau(z) integrated from xe(z)
    Params:
        xe : ionization rate of the Universe
        z : list in descending order (from 30 to 0 for instance)
    """
    z = np.sort(z)
    xe = np.sort(xe)[::-1]
    csurH = c*100/cos.H(0).si.value  #cm
    Hdezsurc = cos.H(z).si.value/(c*100) #cm-1

    eta=1 #eta=1 for now
    integ2 = c * s_T * nh * xe / cos.H(z).si.value * (1+z)**2 * (1+eta*Yp/4/Xp)
    taudez2 = cumtrapz(integ2, z, initial=0)

    return taudez2


def xe_asym2(z, H_zend=5.5, alpha=8.4, z_early=20.,
                helium1=True, He_zend=5.5,
                helium2=False, helium2_redshift=3.5, helium2_deltaredshift=.5,
                 xe_recomb=1.7e-4):
    """
    Computes the redshift-asymmetric parameterisation of xe(z) in Douspis+2015
    Parameters:
        zend : redshift at wich reionisation ends
        zre : midpoint (when xe = 0.50)
        z_early : redshift aroiund which the first sources form (taken to 20)
        helium, helium2 : to include helium first and second reionisation or not
        xe_recomb : ionised fraction leftover after recombination (default is 1.7e-4)
        CAREFUL z must not go further than z_early

    """
    fH = 1.
    frac=0.5*(np.sign(H_zend-z)+1) + 0.5*(np.sign(z-H_zend)+1)*abs((z_early-z)/(z_early-H_zend))**(alpha)

    xe = (fH-xe_recomb)*frac +xe_recomb

    if helium1:
        fHe = Yp/(3.9715*(1-Yp))
        frac=0.5*(np.sign(He_zend-z)+1) + 0.5*(np.sign(z-He_zend)+1)*abs((z_early-z)/(z_early-He_zend))**(alpha)

        xe += (fHe-xe_recomb)*frac + xe_recomb
    if helium2:
        assert helium1, "Need to set both He reionisation to True, cannot have HeII without HeI"
        a = np.divide(1,z+1.)
        deltayHe2 = 1.5*np.sqrt(1+helium2_redshift)*helium2_deltaredshift
        VarMid2    = (1.+helium2_redshift)**1.5
        xod2 = (VarMid2 - 1./a**1.5)/deltayHe2
        tgh2 = np.tanh(xod2) # check if xod<100
        xe += (fHe-xe_recomb) * (tgh2+1.)/2.
    return xe
