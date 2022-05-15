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

for ac in ['tanh','DReLU']:
	out=[]
	for n in n_:
		fn=ac+' n='+str(n)+' depth='+str(depth)+' scaling='+scaling
		AS=jnp.squeeze(bk.get('outputs/AS '+fn))
		NS=jnp.squeeze(bk.get('outputs/NS '+fn))
		rel=avgsq(AS,axis=-1)/avgsq(NS,axis=-1)
		out.append([jnp.quantile(rel,.25),jnp.median(rel),jnp.quantile(rel,.75)])
	out=jnp.array(out)
	plt.plot(n_,out[:,1],color=colors[ac])
	plt.fill_between(n_,out[:,2],out[:,0],color=colors[ac],alpha=.1)

plt.yscale('log')
plt.show()

