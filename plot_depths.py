import jax.numpy as jnp
from util import str_
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
import os

def avgsq(x):
	return jnp.average(jnp.square(x))


def instancemeans(ac,depth,n,scaling):
	path=str_('outputs/depth=',depth,' AS/',ac,' n=',n,' scaling=',scaling,'/')
	return jnp.squeeze(jnp.array([avgsq(bk.get(path+i)) for i in os.listdir(path)]))

def getinstances(ac,depth,n_,scaling):
	return [instancemeans(ac,depth,n,scaling) for n in n_]
#
#
#def get_averages(depth,ns,scaling):
#
#	E_AS=dict()
#	E_NS=dict()
#	for ac in ns.keys():
#		fns_AS=['outputs/depth='+str(depth)+' AS/'+ac+' n='+str(n)+' scaling='+scaling+'/' for n in ns[ac]]
#		fns_NS=['outputs/depth='+str(depth)+' NS/'+ac+' n='+str(n)+' scaling='+scaling for n in ns[ac]]
#		E_AS[ac]=[getinstanceaverages(fn) for fn in fns_AS]
#		E_NS[ac]=[avgsq(jnp.squeeze(bk.get(fn)),axis=-1) for fn in fns_NS]
#
#	return E_AS,E_NS


def plot(ax,depth,ac,ns,color,**kwargs):
	instances=getinstances(ac,depth,ns,scaling)
	ax.plot(ns,[jnp.average(I) for I in instances],color=color,**kwargs)
	quartiles=[[jnp.quantile(I,q) for I in instances] for q in [1/4,1/2,3/4]]
	ax.fill_between(ns,quartiles[0],quartiles[-1],color=color,alpha=.2)




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
		
		plot(ax_[i],depth,'DReLU_normalized',n_,color='blue',label='DReLU')
		plot(ax_[i],depth,'tanh',n_,color='red',label='tanh')

		#E_AS,E_NS=get_averages(depth,{ac:n_ for ac in acs},scaling)
		#ax_[i].plot(n_,jnp.median(E_NS['DReLU'],axis=1),color='blue',lw=1,label=r'$f$')
		#ax_[i].plot(n_,jnp.median(E_NS['tanh'],axis=1),color='red',ls='dotted',lw=2,label=r'$f$')
		#ax_[i].plot(n_,jnp.median(E_AS['DReLU'],axis=1),color='blue',marker='o',ms=4,label=r'$\mathcal{A}f$ DReLU')
		#ax_[i].plot(n_,jnp.median(E_AS['tanh'],axis=1),color='red',marker='o',ms=4,ls='dashed',label=r'$\mathcal{A}f$ tanh')

		#ax_[i].plot(n_,jnp.median(E_NS['DReLU_normalized'],axis=1),color='blue',lw=.5)
		#ax_[i].plot(n_,jnp.median(E_NS['tanh'],axis=1),color='red',ls='dotted',lw=2)

		#ax_[i].plot(n_,jnp.median(E_AS['DReLU_normalized'],axis=1),color='blue',marker='o',ms=4,label=r'DReLU')
		#ax_[i].plot(n_,jnp.median(E_AS['tanh'],axis=1),color='red',marker='o',ms=4,ls='dashed',label=r'tanh')


		ax_[i].title.set_text(str(depth)+' layers')
		ax_[i].legend()
		ax_[i].set_xlabel(r'$n$')
		if i==0:
			ax_[i].set_ylabel(r'$||\mathcal{A}f||_{\rho}^2$')
		ax_[i].set_xticks(range(2,nmax+1,2))

		ax_[i].set_yscale('log')

	fn='depths='+str(depths)+' scaling='+scaling
	plt.savefig('plots/'+fn+'.pdf',bbox_inches='tight')

