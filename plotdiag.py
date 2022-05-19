import jax
import jax.numpy as jnp
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def getdata(n,d):
	datapath='matlab/Ds/n='+str(n)+'/instance=1.mat'
	data=loadmat(datapath)

	thetas,diag,offdiag,W=[np.squeeze(x) for x in [data['ts'],data['diag'],data['offdiag'],data['W']]]
	t_half=2*thetas[jnp.min(jnp.array(jnp.where(diag>.5)))]

	tmax=5*t_half
	keep=jnp.array(jnp.where(thetas<tmax))

	return [jnp.squeeze(x) for x in [thetas[keep],diag[keep],offdiag[keep],t_half,W]]


def plotdiag(n):
	thetas,diag,_,t_half,_=getdata(100,3)
	tmax=2*t_half

	thetas=jnp.concatenate([-jnp.flip(thetas),thetas])
	diag=jnp.concatenate([jnp.flip(diag),diag])

	plt.figure(figsize=(4,2))
	plt.tight_layout()

	plt.plot(thetas,diag,color='b',lw=2,label=r'$D^{(w)}_{\mathcal{N}}(\theta,\theta)$')
	plt.fill_between(thetas,jnp.zeros(diag.size),diag,color='b',alpha=.2)
	plt.xlim(-tmax,tmax)
	plt.title(r'$n=100$')
	plt.xlabel(r'$\theta$')
	plt.legend()


	ax=plt.gca()
	tr=ax.xaxis.get_ticklabels()[0].get_transform()
	ax.xaxis.set_label_coords(tmax,0,transform=tr)


	plt.savefig('plots/diag n='+str(n)+'.pdf',bbox_inches='tight')


plotdiag(100)
