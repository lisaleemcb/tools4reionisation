#### PARAMETER FILE

# TAU VALUE USED TO CONSTRAIN THE FIT
tau = 0.054
sigtau = 0.008

# COSMOLOGY
cc       = 3e10   #cm s-1
sigt     = 0.666e-24 #cm2
Yp       = 0.2453
eta      = 1
Xp       = 1-Yp
mh       = 1.67e-24 #gr
Om       = 0.309
Ol       = 0.691
h        = 0.6774
rhoc = 1.879e-29 *h**2 #gr cm-3
omb  = 0.02230/h**2
nh = Xp*omb*rhoc/mh  # cm-3

# REIONISATION PARAMETERS
Ch2      = 3.
fesc_xsi = 0.2*10.**53.14
# reionisation of Helium
HeliumI = True
HeliumII = False
fH = 1.
if HeliumI:
	not4 = 3.9715 #eta
	fHe = Yp/(not4*(1-Yp))
	fH=1+fHe
		
# INITIAL SET OF PARAMETERS FOR THE FIT
theta_init = (0.01376,3.26,2.59,5.68)
y0 = 0.0001 #leftover ionised fraction after recombination

