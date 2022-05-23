import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
from util import str_
import jax
import util
import os

def avgsq(x,**kwargs):
	return jnp.average(jnp.square(x),**kwargs)
	

def get_avg(depth,ac,ns,scaling,prefix=''):

	key=jax.random.PRNGKey(1)

	#return [avgsq(bk.get(str_(prefix+'zipoutputs/depth=',depth,' AS/',ac,' n=',n,' ',scaling))) for n in ns]
	#return [avgsq(bk.get(str_(prefix+'/depth=',depth,' AS/',ac,' n=',n,' ',scaling))) for n in ns]
	outputs=[bk.get(str_(prefix+'/depth=',depth,' AS/',ac,' n=',n,' ',scaling)) for n in ns]
	squares=[jnp.square(o) for o in outputs]

	avgsquares=[jnp.average(sq) for sq in squares]
	bootstrapmeans=[util.bootstrapmeans(key,sq) for sq in squares]
	bootstrapquartiles=[[jnp.quantile(means,p) for means in bootstrapmeans] for p in [.05,1/2,.95]]

	return avgsquares,bootstrapquartiles




if __name__=='__main__':

	if len(sys.argv)==1:
		print('\n\nplot_depths nmax min_depth max_depth scaling=X/H\n\n')
		quit()



	nmax=int(sys.argv[1])
	depths=list(range(int(sys.argv[2]),int(sys.argv[3])+1))
	scaling=sys.argv[4] #input('scaling: ')

	acs={'DReLU_normalized','tanh'}
	n_=range(2,nmax+1)
	plt,ax_=plt.subplots(1,len(depths),figsize=(7,1.6),sharey='row')
	if len(depths)==1:
		ax_=[ax_]

	prefix=input('parent folder: ')


	for i,depth in enumerate(depths):
		#ax_[i].plot(n_,jnp.median(E_NS['DReLU_normalized'],axis=1),color='blue',lw=.5)
		#ax_[i].plot(n_,jnp.median(E_NS['tanh'],axis=1),color='red',ls='dotted',lw=2)

		avg_D,q_D=get_avg(depth,'DReLU_normalized',n_,scaling,prefix=prefix)
		avg_t,q_t=get_avg(depth,'tanh',n_,scaling,prefix=prefix)

		ax_[i].plot(n_,avg_D,color='blue',lw=1,label=r'DReLU')
		#ax_[i].plot(n_,avg_D,color='blue',marker='D',ms=3,lw=1,label=r'DReLU')
		ax_[i].fill_between(n_,q_D[0],q_D[-1],color='blue',alpha=.2)

		ax_[i].plot(n_,avg_t,color='red',lw=1,ls='dashed',label=r'tanh')
		#ax_[i].plot(n_,avg_t,color='red',marker='.',ms=2,lw=1,ls='dashed',label=r'tanh')
		ax_[i].fill_between(n_,q_t[0],q_t[-1],color='red',alpha=.2)


		ax_[i].title.set_text(str(depth)+' layers')
		ax_[i].legend()
		ax_[i].set_xlabel(r'$n$')
		if i==0:
			ax_[i].set_ylabel(r'$||\mathcal{A}f||_{\rho}^2$')
		ax_[i].set_xticks(range(2,nmax+1,2))

		ax_[i].set_yscale('log')

	fn='zip depths='+str(depths)+' scaling='+scaling
	plt.savefig('plots/'+fn+'.pdf',bbox_inches='tight')

