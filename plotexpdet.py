import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
import seaborn as sns
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import optax
import util
import spherical
import cancellation as canc



to_Gram=jax.vmap(lambda w:jnp.dot(w,w.T))

def cut(A,B):
	return jnp.dot(B.T,jnp.dot(A,B))

key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)
d=3

instances=50000



"""
ests=[]
for n in n_:
	n=int(n)
	W=jax.random.normal(keys[n],shape=(10000,n,d))/jnp.sqrt(n*d)
	G=jax.vmap(lambda w:jnp.dot(w,w.T))(W)
	squared_est=jnp.exp(1)*jnp.average(jnp.linalg.det(jnp.exp(G)))
	ests.append(jnp.sqrt(squared_est))

ests2=[]
for n in n_:
	n=int(n)
	W=jax.random.normal(keys[n],shape=(50000,n,d))/jnp.sqrt(n*d)
	G=jax.vmap(lambda w:jnp.dot(w,w.T))(W)
	M=jnp.square(G)/2+G+1
	squared_est2=jnp.exp(1)*jnp.average(jnp.linalg.det(M))
	ests2.append(jnp.sqrt(squared_est2))
plt.plot(n_,ests2,'r:')
"""

#log_est=-jnp.multiply((n_-d-1),jnp.log(2*jnp.square(n_)))-d*jnp.log(d)+jnp.log(n_)+1
#plt.plot(n_,jnp.sqrt(jnp.exp(log_est)),'k:')



def Wishart_block_estimate(n):
	W=jax.random.normal(keys[2*n],shape=(instances,n,d))/jnp.sqrt(n*d)
	G=to_Gram(W)

	ones=jnp.ones(shape=(instances,n,1,))
	ones_W=jnp.concatenate([ones,W],axis=-1)
	frame,_=jnp.linalg.qr(ones_W,mode='complete')
	V1=jnp.take(frame,jnp.arange(d+1),axis=-1)
	V2=jnp.take(frame,jnp.arange(d+1,n),axis=-1)

	Ds=jax.vmap(jnp.diag)(to_Gram(jnp.swapaxes(ones_W,-2,-1)))
	det_first=jnp.product(Ds,axis=-1)

	det_second=jnp.linalg.det(jax.vmap(cut,in_axes=(0,0))(jnp.square(G),V2)/2)

	dets=jnp.multiply(det_first,det_second)
	return jnp.sqrt(jnp.exp(1)*jnp.average(dets))	

def rotating_frame_estimate(n):
	det_first=n*jnp.power(1/d,d)

	W=jax.random.normal(keys[2*n],shape=(instances,n,d))
	ones=jnp.ones(shape=(instances,n,1,))
	ones_W=jnp.concatenate([ones,W],axis=-1)
	frame,_=jnp.linalg.qr(ones_W,mode='complete')
	V1=jnp.take(frame,jnp.arange(d+1),axis=-1)
	V2=jnp.take(frame,jnp.arange(d+1,n),axis=-1)
	P1=to_Gram(V1)
	det_second=jnp.linalg.det(jax.vmap(cut,in_axes=(0,0))(jnp.square(P1),V2)/2)

	dets=jnp.multiply(det_first,det_second)
	return jnp.sqrt(jnp.exp(1)*jnp.average(dets))	



plt.figure()
plt.yscale('log')
n_,vals=bk.getplotdata('abbrev/exp_long')
n_=jnp.arange(2,11)
plt.scatter(n_,vals[jnp.arange(len(n_))],color='b')
n_=jnp.arange(4,11)
plt.plot(n_,[Wishart_block_estimate(n) for n in n_],'r')
plt.savefig('plots/expdetplot_goodapprox.pdf')




#plt.plot(n_,[rotating_frame_estimate(n) for n in n_],'r:')
#plt.savefig('plots/expdetplotCF.pdf')




