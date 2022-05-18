import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
import os

def avgsq(x,**kwargs):
	return jnp.average(jnp.square(x),**kwargs)
	

def get_avg(depth,ac,ns,scaling):
	return [avgsq(n,bk.get(str_('zipoutputs/depth=',depth,' AS/',ac,' n=',n,' ',scaling))) for n in ns]


#def getinstanceaverages(path):
#	return jnp.array([avgsq(bk.get(path+f)) for f in os.listdir(path)])
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
		#E_AS,E_NS=get_averages(depth,{ac:n_ for ac in acs},scaling)
		#ax_[i].plot(n_,jnp.median(E_NS['DReLU'],axis=1),color='blue',lw=1,label=r'$f$')
		#ax_[i].plot(n_,jnp.median(E_NS['tanh'],axis=1),color='red',ls='dotted',lw=2,label=r'$f$')
		#ax_[i].plot(n_,jnp.median(E_AS['DReLU'],axis=1),color='blue',marker='o',ms=4,label=r'$\mathcal{A}f$ DReLU')
		#ax_[i].plot(n_,jnp.median(E_AS['tanh'],axis=1),color='red',marker='o',ms=4,ls='dashed',label=r'$\mathcal{A}f$ tanh')

		#ax_[i].plot(n_,jnp.median(E_NS['DReLU_normalized'],axis=1),color='blue',lw=.5)
		#ax_[i].plot(n_,jnp.median(E_NS['tanh'],axis=1),color='red',ls='dotted',lw=2)
		ax_[i].plot(n_,jnp.median(E_AS['DReLU_normalized'],axis=1),color='blue',marker='o',ms=4,label=r'DReLU')
		ax_[i].plot(n_,jnp.median(E_AS['tanh'],axis=1),color='red',marker='o',ms=4,ls='dashed',label=r'tanh')


		ax_[i].title.set_text(str(depth)+' layers')
		ax_[i].legend()
		ax_[i].set_xlabel(r'$n$')
		if i==0:
			ax_[i].set_ylabel(r'$||\mathcal{A}f||_{\rho}^2$')
		ax_[i].set_xticks(range(2,nmax+1,2))

		ax_[i].set_yscale('log')

	fn='depths='+str(depths)+' scaling='+scaling
	plt.savefig('plots/'+fn+'.pdf',bbox_inches='tight')

