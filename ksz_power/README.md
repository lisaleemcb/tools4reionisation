# ksz_power

Here you will find all you need to compute the kSZ angular power spectrum theoretically.


This code is based on Gorce+2020 (arXiv:2004.06616) which presents a simple parameterisation of the patchy kSZ angular power spectrum in terms of reionisation history and morphology. You will need the value of 4 parameters, that will be specified in the parameters.py file:

	- zend, the redshift at which reionisation ends
	
	- zre, the midpoint of EoR (where x_HII = 0.50)
	
	- alpha0, a measure of the variance in the ionisation field (typically, alpha0 = 10 ** 3.5 - 10 ** 4.5 Mpc3)
	
	- kappa, a measure of the typical bubble size during reionisation (typically, kappa = 0.05 - 0.15 Mpc-1)

The computation of the homogeneous part of the kSZ signal is based on Shaw+2012, recalibrated to new simulations.


You only need to run
	run main.py 
to compute the spectrum on a range of multipoles spectified in parameters.py
	run plot.py
to plot the resulting power spectrum (patchy+homogeneous) in muK2.
Currently, < 1 min are needed to compute one C_ell with float32 precision.


The params.py file allows you to fix all the parameters used in the computation, that is:
	- cosmological parameters
	- reionisation parameters (mentioned above)
	- integration and computation parameters, such as the range of multipoles you need the kSZ power at. Note that the values used here are optimised to get the most precise result in the shortest time.

The functions.py file contains the function parameterising the reionisation history (uses the asymmetric parameterisation of Douspis+2015) and can be changed.
The parameterisation used for P_ee(k,z) is also included in this file.


This package requires camb (pycamb) to be installed (https://camb.info) as well as astropy (https://www.astropy.org).

Copyrights: Stephane Ilic & Adelie Gorce
