import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
import plot_depths



def makeplot(nmax,scaling):
	plt.figure(figsize=(7,3))

	ns={'HS':jnp.arange(2,nmax+1),'ReLU':jnp.arange(2,nmax+1),'tanh':jnp.arange(2,nmax+1),'exp':jnp.arange(2,min(8,nmax)+1),}
	acs=[k for k in ns.keys()]

	colors={'exp':'magenta','tanh':'red','ReLU':'blue','HS':'green'}
	ls_={'exp':'dashed','tanh':'solid','ReLU':'solid','HS':'dashed'}
	lw_={'exp':1,'tanh':2,'ReLU':2,'HS':1}
	m_={'exp':'.','tanh':'.','ReLU':'D','HS':'D'}

	E_AS,E_NS=plot_depths.get_averages(2,ns,scaling)

	for ac in acs:
		plt.plot(ns[ac],[jnp.average(ins) for ins in E_AS[ac]],label=ac,color=colors[ac],lw=1,ls=ls_[ac],marker=m_[ac],ms=4)


	Nmax=15
	ns_int={'ReLU':jnp.arange(12,Nmax+2),'HS':jnp.arange(12,Nmax+2)}
	for ac,n_ in ns_int.items():
		integrals=[bk.get('computed_by_iintegral/'+ac+' n='+str(n)) for n in n_]
		plt.plot(n_,[jnp.average(I) for I in integrals],ls='dashed',dashes=(1,2),lw=1,color=colors[ac])


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
