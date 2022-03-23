import numpy as np
import math
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import util
import scratchwork as sc



def Znorm(key,activation,W,X):
	z=jax.random.normal(key,shape=(10000,))
	return util.L2norm(activation(z))

def OPnorm(key,activation,W,X):
	n,d=W.shape[-2:]
	x=util.sample_mu(n*d,10000,key)
	return util.L2norm(activation(x))

def polyOPnorm(key,activation,W,X):
	n,d=W.shape[-2:]
	x=util.sample_mu(n*d,10000,key)
	a,dist=util.polyfit(x,activation(x),n-2)
	#p=util.poly_as_function(a)
	#r=lambda x:activation(x)-p(x)
	return dist

def polyZnorm(key,activation,W,X):
	n=W.shape[-2]
	z=jax.random.normal(key,shape=(10000,))
	a,dist=util.polyfit(z,activation(z),n-2)
	#p=util.poly_as_function(a)
	#r=lambda x:activation(x)-p(x)
	return dist

def OCPnorm(key,activation,W,X):
	variances=jnp.sum(jnp.square(W),axis=(-2,-1))
	covariances=variances-jnp.square(util.mindist(W))
	return jnp.sqrt(jnp.average(util.variations(key,activation,variances,covariances)))

def polyOCPnorm(key,activation,W,X):
	n=W.shape[-2]
	variances=jnp.sum(jnp.square(W),axis=(-2,-1))
	covariances=variances-jnp.square(util.mindist(W))
	_,dist=util.poly_fit_variations(key,activation,n-2,variances,covariances)
	return util.L2norm(dist)
		
def polyOCP_norm(key,activation,W,X):
	n=W.shape[-2]

	covs,signs=sc.covs(W,3)
	covs=covs+0.00001*jnp.eye(covs.shape[-1])[None,:,:]


	#ijk=util.greedyclosepoints(W,3)
	"""
	covs_signs=[util.covs(W[i],ijk[i]) for i in range(W.shape[0])]
	covs=jnp.array([covs_signs[i][0] for i in range(W.shape[0])])
	covs=covs+0.0001*jnp.eye(covs.shape[-1])[None,:,:]
	signs=jnp.array([covs_signs[i][1] for i in range(W.shape[0])])
	"""
	#key1,key2=jax.random.split(key)
	#variances=jnp.sum(jnp.square(W),axis=(-2,-1))
	#covariances=variances-jnp.square(util.mindist(W))
	#coefficients,_=util.poly_fit_variations(key1,activation,n-2,variances,covariances)

	#f=util.polys_as_parallel_functions(coefficients)
	#df=lambda x:activation(x)-f(x)
	#df=activation
	#variations=util.generalized_variations(key2,df,covs,signs)
	#return jnp.sqrt(jnp.average(variations))

	a,dist=util.poly_fit_generalized_variations(key,activation,n-2,covs,signs)
	return util.L2norm(dist)
	
	
def polyOCP_proxynorm(key,activation,W,X):
	n,d=W.shape[-2:]
	key1,key2=jax.random.split(key)
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))

	x=util.sample_mu(n*d,10000,key1)
	a,dist=util.polyfit(x,activation(x),n-2)
	p=util.poly_as_function(a)
	r=lambda x:activation(x)-p(x)


"""
activation-specific proxies
"""
def exactexp(W,X):
	n=W.shape[-2]
	nfactor=1.0
	for k in range(1,n+1):
		nfactor=nfactor/jnp.sqrt(k)
	instances_samples_n_n=jnp.swapaxes(jnp.exp(jnp.inner(W,X)),1,2)
	return nfactor*jnp.linalg.det(instances_samples_n_n)

def tanhtaylor(n_):
	nmax=20
	a=[1.0]
	N=int(input('nmax'))
	for n in range(N):
		s=0
		for k in range(n+1):
			s=s+a[k]*a[n-k]	
		a.append(s/(2*n+3))
	return jnp.array(a)[n_]
