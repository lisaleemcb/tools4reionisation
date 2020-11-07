import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate

from parameters import *

plt.ion()

fig = plt.figure(figsize=(9,8))
ax=fig.add_axes([0.1, 0.12, 0.85, 0.85])
ax.set_xlabel(r'Angular multipole $\ell$',fontsize=16)
ax.set_ylabel(r'$\mathcal{D}_\ell^{kSZ}$ [$\mu$K$^2$]',fontsize=16)
ax.set_xlim(0,1e4)
ax.set_ylim(0,5.)
ax.tick_params(labelsize=14)

######## results

out = np.loadtxt(outroot+'_kSZ_Cells.txt')
l1 = out[:,0]
PS1 = l1*(l1+1)*out[:,1]/(2*np.pi)*(T_CMB_uK**2) # 
plt.plot(l1,PS1,color='C0',lw=1.2,ls='--',label=r'Patchy signal')
if late_time:
	PS2 = l1*(l1+1)*(out[:,2]-out[:,1])/(2*np.pi)*(T_CMB_uK**2)
	PS = PS1 + PS2
	plt.plot(l1,PS2,color='C0',lw=1.2,ls='-.',label=r'Homogeneous signal')
	plt.plot(l1,PS1+PS2,color='C0',lw=1.5,label='Total signal')


######## data
plt.errorbar(3000,1.1,yerr=[[0.7],[1.0]],lw=0.,capsize=3.,elinewidth=1.2,color='k',marker='o',ms=5,alpha=.7) #reichardt+2020

plt.legend(loc='upper left',frameon=False,fontsize=16,ncol=1)
# plt.savefig('./kSZ_PS.png')

print('Results:')
ls = np.linspace(min(l1),max(l1),10000)
if late_time:
	homo = interpolate.interp1d(l1,PS2)
	print(' homogeneous... %.2f muK^2' %(homo(3000)))
patchy = interpolate.interp1d(l1,PS1)
print(' patchy... %.3f muK^2' %(patchy(3000)))
pps = patchy(ls)
print(' lmax... %.1f' %(ls[np.argmax(pps)]))
print(' max ampl... %.2f ' %(np.max(pps)))
