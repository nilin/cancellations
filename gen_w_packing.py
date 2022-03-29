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
import plot_W_delta 


#seed=int(input('input seed for randomness '))
seed=0
nmax=20
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





def HCP_lattice(n):
	v1=jnp.array((1,0,0))
	v2=jnp.array((1/2,jnp.sqrt(3)/2,0))
	v3=jnp.array((1/2,1/(2*jnp.sqrt(3)),jnp.sqrt(2/3)))

	I=jnp.arange(-n,n)
	X1=I[:,None]*v1[None,:]
	X2=I[:,None]*v2[None,:]
	X3=I[:,None]*v3[None,:]
	
	L=X1[:,None,None,:] +X2[None,:,None,:] +X3[None,None,:,:]

	return jnp.reshape(L,(I.size**3,3))


def restrict(L,n):
	sqdists=jnp.sum(jnp.square(L),axis=-1)
	indices=jnp.argsort(sqdists)[:n]
	return L[indices]


def gen_w_separated_3d(n,eps,shifts,key):
	lattice=HCP_lattice(10)
	shiftedL=lattice[None,:,:]+eps*jax.random.normal(key,(shifts,3))[:,None,:]
	L=jnp.array([restrict(shiftedL[i],n) for i in range(shifts)])
	norms=jnp.sqrt(jnp.sum(jnp.square(L),axis=(-2,-1)))
	i=jnp.argmin(norms)
	return L[i]/norms[i]





#centers=gen_w_separated_3d(10,key0)	
#
#print(jnp.sum(jnp.square(centers)))
#
#radii=jnp.ones(centers.shape[0])/2
#
#plot_W_delta.plot_packing(centers)




def gen(eps):
	ws={n:gen_w_separated_3d(n,eps,25,keys[n]) for n in n_}
	deltas={n:util.mindist(ws[n]) for n in n_}
	bk.savedata({'ws':ws,'n_':n_,'deltas':deltas},'w_packing')
	
	plt.plot(n_,[deltas[n] for n in n_],'r')
	

def compare():
	Ws={n:util.normalize(jax.random.normal(keys[n],(instances,n,d))/jnp.sqrt(n*d)) for n in n_}
	deltas={n:util.L2norm(util.mindist(Ws[n])) for n in n_}
	plt.plot(n_,[deltas[n] for n in n_],'b')



density=math.pi/(3*jnp.sqrt(2))

plt.plot(n_,[2*density**(1/3)*(5/3)**(1/2)*n**(-5/6) for n in n_],'m:')

gen(1)
compare()
plt.yscale('log')
plt.show()

"""
def gen():
	Ws={n:gen_W_separated(keys[n],instances,n,d) for n in n_}
	deltas={n:util.L2norm(util.mindist(Ws[n])) for n in n_}
	bk.savedata({'Ws':Ws,'Wtype':'separated','instances':instances,'d':d,'n_':n_,'deltas':deltas,'seed':seed},'W_separated, seed='+str(seed))
	plt.plot(n_,[deltas[n] for n in n_])



gen()
compare()
plt.yscale('log')
plt.show()
"""
