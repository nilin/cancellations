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
import cancellation as canc




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
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))
	return jnp.sqrt(jnp.average(util.variations(key,activation,r_squared,eps_squared)))

def polyOCPnorm(key,activation,W,X):
	n=W.shape[-2]
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))
	key1,key2=jax.random.split(key)
	a,dist=util.poly_fit_variations(key1,activation,n-2,r_squared,eps_squared)
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
