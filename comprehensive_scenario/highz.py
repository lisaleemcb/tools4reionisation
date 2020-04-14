# import modules for solving 
import scipy.integrate
import highz_config as hz

def sfrd(z,theta):         
    """ Computes star formation history according to chosen parametrisation """
    ap,bp,cp,dp         = theta 
    rhosfr              = ap*(1+z)**bp/(1+((1.+z)/cp)**dp)
    return rhosfr

def nion(z,theta):         # specify the initial time point t0=0
    """ Computes ionising emissivity from star formation history """
    rhosfr  = sfrd(z,theta)/(1./31557e12*(3.0856775813057E+24)**3)
    nion    = (hz.fesc_xsi*(rhosfr)/hz.nh)
    return nion


def dQHIIdz(y, z,a,b,c,d):         # specify the initial time point t0=0
    """ Gives differential eq. for QHII """
    theta=(a,b,c,d)
    trec                  = 3.19e2*(1+z)**(-3.) * (3./hz.Ch2)#ch2=3
    H0m1                  = 9.77*hz.h**(-1) #9.77*h^-1 Gyr
    dtdz                  = -H0m1*1./(1+z)*(hz.Om*(1+z)**3.+hz.Ol)**(-0.5)
    dQHIIdz   = (nion(z,theta)-(y/trec))*dtdz
    return dQHIIdz

def QHII(z,y0,theta):
    """ Solves diff eq. to derive QHII(z) """
    xe = scipy.integrate.odeint(dQHIIdz, y0, z,args =theta).flatten()
    xe[xe>hz.fH] = hz.fH
    return xe




