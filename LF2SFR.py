import numpy as np 
from astropy import cosmology as cos 
import mpmath
import astropy.units as units
from scipy.integrate import simps

#cosmology
p15 = cos.Planck15

# conversion magnitude to luminosity
def M2L(M,z):
    bouwens_dmodulus = p15.distmod(z).value-2.5*np.log10(1.+z)
    l_dist = p15.luminosity_distance(z).to(units.cm).value
    mstarapp = M+bouwens_dmodulus
    lstar = ((10**(-0.4*(48.60+mstarapp)))*4.*np.pi*(l_dist**2))/(1+z)  
    return lstar

# conversion luminosity to magnitude
def L2M(L,z):
    bouwens_dmodulus = p15.distmod(z).value-2.5*np.log10(1.+z)
    l_dist = p15.luminosity_distance(z).to(units.cm).value
    mstarapp = -2.5*np.log10( (1.+z)*L / 4. / np.pi / l_dist**2 ) - 48.6
    M = mstarapp + bouwens_dmodulus
    return M 

def Schechter_LF(M,z,phi_star,M_star,alpha,lum=False):

    L = M2L(M,z)
    if (lum):
        M_star = L2M(M_star)
    #distance modulus
    bouwens_dmodulus = p15.distmod(z).value-2.5*np.log10(1.+z)
    #luminosity distance for this redshift
    l_dist = p15.luminosity_distance(z).to(units.cm).value
    # number density Number/mag/Mpc^3
    n_M = phi_star * np.log(10)*0.4 * 10**( -0.4*(M-M_star)*(alpha+1.) ) * np.exp(- 10**( -0.4*(M-M_star) ) )
    return n_M

def UV_density(M,z,Mlim,phi_star,M_star,alpha,lum=False)

    L = M2L(M,z)
    Llim = M2L(Mlim,z)
    Lstar = M2L(M_star,z)
    if (lum):
        M_star = L2M(M_star,z)
        Mlim = L2M(Mlim,z)
    n_M = Schechter_LF(M,z,Mlim,phi_star,M_star,alpha,lum)
    # UV luminosity density, ergs.s-1.Hz-1.Mpc-3 
    rho= phi_star*Lstar*mpmath.fp.gammainc(alpha+2., Llim, b = np.inf) #analytical way to derive UV density
    # rho=simps(n_M[M<Mlim*L[M<Mlim],M[M<Mlim) #numerical way of deriving UV density
    return rho

def SFR_density(M,z,Mlim,phi_star,M_star,alpha,lum=False)

    L = M2L(M,z)
    Llim = M2L(Mlim,z)
    Lstar = M2L(M_star,z)
    if (lum):
        M_star = L2M(M_star,z)
        Mlim = L2M(Mlim,z)
    n_M = Schechter_LF(M,z,Mlim,phi_star,M_star,alpha,lum)
    # UV luminosity density, ergs.s-1.Hz-1.Mpc-3 
    rho= phi_star*Lstar*mpmath.fp.gammainc(alpha+2., Llim, b = np.inf) #analytical way to derive UV density
    # rho=simps(n_M[M<Mlim*L[M<Mlim],M[M<Mlim) #numerical way of deriving UV density
    SFR2UV = 8.0e27 #egrs s-1 Hz-1 Madau 1998
    SFR = rho - np.log10(SFR2UV) 
    return SFR



