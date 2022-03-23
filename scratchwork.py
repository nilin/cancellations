import permutations#
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
import sys
import jax
import jax.numpy as jnp
import optax
import util
import cancellation as canc

###


def covs(W,m):
	instances=W.shape[0]
	n=W.shape[-2]
	p=list(range(m))
	s=1
	pWlist=[]
	signs=[]
	for i in range(math.factorial(m)):
		p_=permutations.embed(p,list(range(m)),n)
		P=permutations.perm_as_matrix(p_)
		PW=util.applymatrixalongdim(P,W,-2)
		pWlist.append(jnp.reshape(PW,(instances,-1))[:,None,:])
		signs.append(s)

		p,ds=permutations.nextperm(p)
		s=s*ds

	I_P_nd=jnp.concatenate(pWlist,axis=-2)
	covs=jax.vmap(jnp.inner,in_axes=(0,0))(I_P_nd,I_P_nd)

	return covs,jnp.array(signs)


#
#def covs_(w,indices):
#	indices=list(indices)
#	m=len(indices)
#	n=w.shape[0]
#	p=list(range(m))
#	s=1
#	pwlist=[]
#	signs=[]
#	nperms=math.factorial(m)
#	for i in range(nperms):
#		p_=permutations.embed(p,indices,n)
#		P=permutations.perm_as_matrix(p_)
#		Pw=applymatrixalongdim(P,w,-2)
#		pwlist.append(Pw)
#		signs.append(s)
#
#		p,ds=permutations.nextperm(p)
#		s=s*ds
#	pwarray=jnp.reshape(jnp.array(pwlist),(nperms,-1))
#	covs=jnp.inner(pwarray,pwarray)	
#	return covs,jnp.array(signs)
#	
	
	

def greedyclosepoints(W,L): #slow proof of concept
	instances=W.shape[0]
	ks=[]
	ij=util.argmindist(W)
	for I in range(instances):
		kI=list(ij[I])
		for l in range(3,L+1):
			w=W[I]
			i,j=ij[I][0],ij[I][1]
			costs=jnp.max(jnp.array([jnp.sum(jnp.square(w[:,:]-w[i,:]),axis=-1)+jnp.sum(jnp.square(w[:,:]-w[j,:]),axis=-1)]),axis=0)
			k=0
			while k==i or k==j:
				k=k+1
			for q in range(w.shape[0]):
				if q!=i and q!=j and costs[q]<costs[k]:
					k=q
					
			kI.append(k)
		ks.append(kI)
	return jnp.array(ks)


def greedycloseordering(W):
	n=W.shape[-2]
	print(n)
	particleorderings=greedyclosepoints(W,n)	
	right_mult_P=jax.vmap(permutations.perm_as_matrix)(particleorderings)
	P=jnp.swapaxes(right_mult_P,-2,-1)
	W_ordered=jax.lax.batch_matmul(P,W)
	return W_ordered
	

		
