import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
import plot_depths
from util import str_




def avgsq(x):
	return jnp.average(jnp.square(x))


def instancemeans(ac,n,scaling,prefix=''):
	path=str_(prefix+'outputs/depth=',2,' AS/',ac,' n=',n,' ',scaling,'/')
	return jnp.squeeze(jnp.array([avgsq(bk.get(path+i)) for i in os.listdir(path)]))

def getinstances(ac,n_,scaling,prefix=''):
	return [instancemeans(ac,n,scaling,prefix) for n in n_]



def makeplot(nmax,scaling):


	prefix=input('parent folder: ' )


	plt.figure(figsize=(7,3))

	ns={'HS':jnp.arange(2,nmax+1),'ReLU':jnp.arange(2,nmax+1),'tanh':jnp.arange(2,nmax+1),'exp':jnp.arange(2,min(9,nmax)+1),}
	acs=[k for k in ns.keys()]

	colors={'exp':'magenta','tanh':'red','ReLU':'blue','HS':'green'}
	ls_={'exp':'dashed','tanh':'solid','ReLU':'solid','HS':'dashed'}
	lw_={'exp':1,'tanh':2,'ReLU':2,'HS':1}
	m_={'exp':'.','tanh':'.','ReLU':'D','HS':'D'}


	for ac in acs:
		instances=getinstances(ac,ns[ac],scaling,prefix)
		#plt.plot(ns[ac],[jnp.average(I) for I in instances],label=ac,color=colors[ac],lw=1,ls=ls_[ac],marker=m_[ac],ms=4)
		quartiles=[[jnp.quantile(I,q) for I in instances] for q in [1/4,1/2,3/4]]
		plt.plot(ns[ac],quartiles[1],label=ac,color=colors[ac],lw=1,ls=ls_[ac],marker=m_[ac],ms=4)
		plt.fill_between(ns[ac],quartiles[0],quartiles[-1],color=colors[ac],alpha=.2)


	Nmax=15
	ns_int={'ReLU':jnp.arange(nmax,Nmax+2),'HS':jnp.arange(nmax,Nmax+2)}
	for ac,n_ in ns_int.items():
		integrals=[bk.get('computed_by_iintegral/'+ac+' n='+str(n)) for n in n_]
		plt.plot(n_,[jnp.median(I) for I in integrals],ls='dashed',dashes=(1,2),lw=1,color=colors[ac])


	plt.legend()
	plt.xlabel(r'$n$')
	plt.ylabel(r'$||\mathcal{A}f||_{\rho}^2$')
	plt.xticks(range(2,Nmax+1))
	plt.yscale('log')
	plt.ylim(bottom=1/10**9)
	plt.xlim(1,Nmax+1)

	fn=str(acs)+' scaling='+scaling
	plt.savefig('plots/'+fn+'.pdf',bbox_inches='tight')



if __name__=='__main__':

	if sys.argv[1]=='h':
		print('\n\nplot_twolayer.py nmax scaling=X/H\n\n');quit()

	nmax=int(sys.argv[1])
	scaling=sys.argv[2]
	makeplot(nmax,scaling)
