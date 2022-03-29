import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import pickle
import time
import bookkeep
import copy
import jax
import jax.numpy as jnp
import optax
import util
import bookkeep as bk


#seed=int(input('input seed for randomness '))
seed=0
nmax=50
d=3
key=jax.random.PRNGKey(seed)
key0,*keys=jax.random.split(key,1000)
n_=range(2,nmax+1)
instances=10000


def gen_W_separated(key,instances,n,d):
	return util.normalize(separate(util.normalize(jax.random.normal(key,shape=(instances,n,d)))))

def separate(W,iterations=1000,optimizer=optax.rmsprop(.01),smoothingperiod=25):
	(instances,n,d)=W.shape
	print('\n'+str(n))
	state=optimizer.init(W)

	losses=[]
	i=0
	while i<100 or loss<losses[i-100]*.999:
		loss,grads=jax.value_and_grad(energy)(W)
		updates,_=optimizer.update(grads,state,W)
		W=optax.apply_updates(W,updates)
		W=util.normalize(W)
		losses.append(loss)
		bk.printbar(loss/losses[0],'')
		i=i+1
	print(i)
	return W

def energies(X):
	eps=.0001
	dists=jnp.sqrt(util.pairwisesquaredists(X)+eps)
	energies=jnp.triu(jnp.exp(-dists),k=1)
	return jnp.max(energies,axis=(-2,-1))

energy=lambda W:jnp.sum(energies(W))#+1000*jnp.sum(jnp.square(W))



def gen():
	Ws={n:gen_W_separated(keys[n],instances,n,d) for n in n_}
	deltas={n:util.L2norm(util.mindist(Ws[n])) for n in n_}
	bk.savedata({'Ws':Ws,'Wtype':'separated','instances':instances,'d':d,'n_':n_,'deltas':deltas,'seed':seed},'W_separated, seed='+str(seed))
	plt.plot(n_,[deltas[n] for n in n_])

def compare():
	Ws={n:util.normalize(jax.random.normal(keys[n],(instances,n,d))/jnp.sqrt(n*d)) for n in n_}
	deltas={n:util.L2norm(util.mindist(Ws[n])) for n in n_}
	plt.plot(n_,[deltas[n] for n in n_])


gen()
compare()
plt.yscale('log')
plt.show()
