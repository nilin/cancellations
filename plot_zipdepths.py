import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
from util import str_
import os

def avgsq(x,**kwargs):
	return jnp.average(jnp.square(x),**kwargs)
	

def get_avg(depth,ac,ns,scaling):
	return [avgsq(bk.get(str_('zipoutputs/depth=',depth,' AS/',ac,' n=',n,' ',scaling))) for n in ns]




if __name__=='__main__':

	if len(sys.argv)==1:
		print('\n\nplot_depths nmax min_depth max_depth scaling=X/H\n\n')
		quit()



	nmax=int(sys.argv[1])
	depths=list(range(int(sys.argv[2]),int(sys.argv[3])+1))
	scaling=sys.argv[4] #input('scaling: ')

	acs={'DReLU_normalized','tanh'}
	n_=range(2,nmax+1)
	plt,ax_=plt.subplots(1,len(depths),figsize=(7,2),sharey='row')
	if len(depths)==1:
		ax_=[ax_]

	for i,depth in enumerate(depths):
		#ax_[i].plot(n_,jnp.median(E_NS['DReLU_normalized'],axis=1),color='blue',lw=.5)
		#ax_[i].plot(n_,jnp.median(E_NS['tanh'],axis=1),color='red',ls='dotted',lw=2)
		ax_[i].plot(n_,get_avg(depth,'DReLU_normalized',n_,scaling),color='blue',marker='o',ms=4,label=r'DReLU')
		ax_[i].plot(n_,get_avg(depth,'tanh',n_,scaling),color='red',marker='o',ms=4,ls='dashed',label=r'tanh')


		ax_[i].title.set_text(str(depth)+' layers')
		ax_[i].legend()
		ax_[i].set_xlabel(r'$n$')
		if i==0:
			ax_[i].set_ylabel(r'$||\mathcal{A}f||_{\rho}^2$')
		ax_[i].set_xticks(range(2,nmax+1,2))

		ax_[i].set_yscale('log')

	fn='depths='+str(depths)+' scaling='+scaling
	plt.savefig('plots/'+fn+'.pdf',bbox_inches='tight')

