import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk
import sys


def avgsq(x,**kwargs):
	return jnp.average(jnp.square(x),**kwargs)
	

nmax=int(sys.argv[1])
depth=int(sys.argv[2]) #int(input('depth: '))
scaling=sys.argv[3] #input('scaling: ')


n_=range(2,nmax+1)
colors={'tanh':'red','DReLU':'blue'}
ls={'tanh':'dashed','DReLU':'solid'}

for ac in ['tanh','DReLU']:
	fns=[ac+' n='+str(n)+' depth='+str(depth)+' scaling='+scaling for n in n_]
	E_AS=jnp.stack([avgsq(jnp.squeeze(bk.get('outputs/AS '+fn)),axis=-1) for fn in fns],axis=0)
	print(E_AS.shape)
	E_NS=jnp.stack([avgsq(jnp.squeeze(bk.get('outputs/NS '+fn)),axis=-1) for fn in fns],axis=0)
	plt.plot(n_,jnp.median(E_AS,axis=1),color=colors[ac],marker='o',ms=3,ls=ls[ac])
	plt.plot(n_,jnp.median(E_NS,axis=1),color=colors[ac],alpha=.2)

plt.yscale('log')
plt.show()

