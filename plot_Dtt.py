import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
import jax.numpy as jnp
import bookkeep as bk
import sys

def plot2d(n,i):
	s,t,D=bk.get('D/forplot/n='+str(n)+'/instance '+str(i))
	tmax=jnp.max(t)

	S,T=jnp.meshgrid(s,t)

	print(D.shape)

	plt.figure(figsize=(2,2))

	cmap=lcm([[1,1+i,1+i] for i in jnp.arange(-1,0,.01)]+[[1-i,1-i,1] for i in jnp.arange(0,1.01,.01)])

	plt.pcolormesh(S,T,D,cmap=cmap,vmin=-1,vmax=1,lw=0,rasterized=True)	
	plt.gca().set_aspect('equal')
	plt.xlim((-tmax,tmax))
	plt.ylim((-tmax,tmax))

	plt.xlabel(r'$\theta$')
	plt.ylabel(r'$\tilde\theta$')

	tmax=round(tmax)

	plt.xticks([1-tmax,0,tmax-1])
	plt.yticks([1-tmax,0,tmax-1])

	plt.title('n='+str(n))

	#cbar=plt.colorbar()
	#cbar.set_ticks([0,1])
	plt.savefig('plots/Dtt n='+str(n)+'.pdf',bbox_inches='tight')

	print('done')

plot2d(sys.argv[1],0)
