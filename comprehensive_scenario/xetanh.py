# Reionisation module
# MD 05/2015 from idl routines from SI and MD


import matplotlib.pyplot as plt
import numpy as np

xe_recomb = 1e-4

helium_fullreion_redshift       = 3.5
helium_fullreion_deltaredshift  = 0.5
helium_fullreion_redshiftstart  = 5.0
yp                              = 0.2453
not4                            = 3.9715 #eta

def xe_tanh(z,ze,deltaz=0.5, helium=True, helium2=False):

     """
     computes the tanh reio with and without Helium part
     as function of z_reio and delta_z
     ze = z of instantaneous reionization
     deltaz = slope/reionization period (0.5)
     """

     fHe       = yp/(not4*(1-yp)) #number of electron per H nucleus (>1 if He ionized)

     deltay    = 1.5*np.sqrt(1+ze)*deltaz
     VarMid    = (1.+ze)**1.5
     a         = 1./(z+1.)

     xod = ((1+ze)**1.5 - (1+z)**1.5)/deltay

     tgh = np.zeros(z.size)+1.0
     tgh = np.tanh(xod)

     xe = (1.-xe_recomb)*(tgh+1.)/2.+xe_recomb

     if (helium):
         xe = ((1.+fHe)-xe_recomb)*(tgh+1.)/2.+xe_recomb

     if (helium2):
         #w=np.where(z < 3.5)
         #xe[w] = ((1.+2*fHe)-xe_recomb)*(tgh[w]+1.)/2.+xe_recomb

         deltayHe2 = 1.5*np.sqrt(1+helium_fullreion_redshift)*helium_fullreion_deltaredshift
         VarMid2    = (1.+helium_fullreion_redshift)**1.5

         xod2 = (VarMid2 - 1./a**1.5)/deltayHe2

         tgh2 = np.zeros(z.size)+1.0
         tgh2 = np.tanh(xod2) # check if xod<100

         xe = xe + ((fHe)-xe_recomb)*(tgh2+1.)/2.


     return xe

def xe_asym(z,zend,zre,z_early=20.,helium=True,helium2=False,xe_recomb=xe_recomb):
    """
    computes the redshift-asymmetric parametrization of xe(z)
    with and without Helium part
    as a function of zend (redshift at which reio ends),
    z_early (redshift around which the first emitting
    sources form) and alpha, the power law
    """
    f=1
    if (helium):
        not4 = 3.9715 #eta
        fHe = yp/(not4*(1-yp))
        f=1+fHe
    alpha = np.log(1./2./f)/ np.log((z_early-zre)/(z_early-zend))
    frac=0.5*(np.sign(zend-z)+1) + 0.5*(np.sign(z-zend)+1)*(((z_early-z)/(z_early-zend))**(alpha)) 
    xe = (f-xe_recomb)*frac +xe_recomb
    if (helium2):
        if (helium==False):
            print("Need to set both He reionisation to True, cannot have HeII without HeI")
            sys.exit()
        helium_fullreion_redshift = 3.5
        helium_fullreion_start = 5.0
        helium_fullreion_deltaredshift = 0.5
        a = np.divide(1,z+1.)
        deltayHe2 = 1.5*np.sqrt(1+helium_fullreion_redshift)*helium_fullreion_deltaredshift
        VarMid2    = (1.+helium_fullreion_redshift)**1.5
        xod2 = (VarMid2 - 1./a**1.5)/deltayHe2
        tgh2 = np.tanh(xod2) # check if xod<100
        xe = (f-xe_recomb)*frac + ((fHe)-xe_recomb)*(tgh2+1.)/2. +xe_recomb 
    return xe


