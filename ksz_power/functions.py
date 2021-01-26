from scipy.integrate import simps, cumtrapz, quad, trapz
from scipy import interpolate
import numpy as np
from astropy import cosmology
from parameters import *
cos=cosmology.FlatLambdaCDM(H0=h*100,Tcmb0=T_cmb,Ob0=Ob_0,Om0=Om_0)

##########################################
######### Reionisation functions #########
##########################################

def xe(z):
    """
    Redshift-evolution of the ionised level of the IGM. 
    You can use a redshift-symmetric (tanh) or redshift-asymmetric (power-law) model.
    You can also enter your own xe(z) here.
    """
    if asym:
        return xe_asym(z,zend,zre)
    else:
        return xe_tanh(z,zre,0.5)

def xe_asym(z,zend=zend,zre=zre,z_early=z_early,helium=HeliumI,helium2=HeliumII,xe_recomb=xe_recomb):
    """
    Computes the redshift-asymmetric parameterisation of xe(z) in Douspis+2015
    Parameters:
        helium, helium2 : to include helium first and second reionisation or not
        zre : midpoint (when xe = 0.50) 
        z_end : redshift at wich reionisation ends
        z_early : redshift aroiund which the first sources form (taken to 20)
        CAREFUL z must not go further than z_early

    """
    alpha = np.log(1./2./fH)/ np.log((z_early-zre)/(z_early-zend))
    frac=0.5*(np.sign(zend-z)+1) + 0.5*(np.sign(z-zend)+1)*(((z_early-z)/(z_early-zend))**(alpha)) 
    xe = (fH-xe_recomb)*frac +xe_recomb
    if (helium2):
        assert helium, "Need to set both He reionisation to True, cannot have HeII without HeI"
        a = np.divide(1,z+1.)
        deltayHe2 = 1.5*np.sqrt(1+helium_fullreion_redshift)*helium_fullreion_deltaredshift
        VarMid2    = (1.+helium_fullreion_redshift)**1.5
        xod2 = (VarMid2 - 1./a**1.5)/deltayHe2
        tgh2 = np.tanh(xod2) # check if xod<100
        xe = (fH-xe_recomb) * ( frac + (tgh2+1.)/2. ) + xe_recomb 
    return xe


def xe_tanh(z,ze=zre,deltaz=0.5, helium=HeliumI, helium2=HeliumII,xe_recomb=xe_recomb):

     """
     computes the redshift-symmetric parameterisation of xe(z) (tanh)
     as function of z_reio (midpoint) and delta_z (duration)
     """
     deltay    = 1.5*np.sqrt(1+ze)*deltaz
     VarMid    = (1.+ze)**1.5

     xod = ((1+ze)**1.5 - (1+z)**1.5)/deltay
     tgh = np.tanh(xod)

     xe = (fH-xe_recomb)*(tgh+1.)/2.+xe_recomb

     if (helium2):
        a = 1./(z+1.)
        deltayHe2 = 1.5*np.sqrt(1+helium_fullreion_redshift)*helium_fullreion_deltaredshift
        VarMid2    = (1.+helium_fullreion_redshift)**1.5
        xod2 = (VarMid2 - 1./a**1.5)/deltayHe2
        tgh2 = np.tanh(xod2)
        xe = xe + (fHe-xe_recomb)*(tgh2+1.)/2.

     return xe

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

# window function (power law) for early times
def W1(k,x):
    if (x<=xe_recomb):
        return 1e-20
    else:
        return  10**alpha0 * x**(-0.2) / (1. + x*(k/kappa)**3. )
W = np.vectorize(lambda k,x: W1(k,x))

# electrons - matter bias after reionisation
def bdH(k,z):
    kf = 9.4
    g = 0.5
    return 0.5 * ( np.exp(-k/kf) + 1. / (1. + np.power(g*k/kf,2.)) )

