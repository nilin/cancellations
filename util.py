import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import pdb
	

@jax.jit
def ReLU(x):
	return jnp.maximum(x,0) 

activations={'ReLU':ReLU,'tanh':jnp.tanh}

@jax.jit
def sqlossindividual(Y,Z):
	Y,Z=[jnp.squeeze(_) for _ in (Y,Z)]
	return jnp.square(Y-Z)

@jax.jit
def sqloss(Y,Z):
	return jnp.average(sqlossindividual(Y,Z))

@jax.jit
def relloss(Y,Z):
	return sqloss(Y,Z)/sqloss(0,Z)


@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))






def randperm(*Xs):
	X=Xs[0]
	n=X.shape[0]
	p=np.random.permutation(n)
	PXs=[np.array(X)[p] for X in Xs]
	#return [jnp.stack([Y[p_i] for p_i in p]) for Y in args]
	return [jnp.array(PX) for PX in PXs]
	

@jax.jit
def apply_on_n(A,X):

	_=jnp.dot(A,X)
	out= jnp.swapaxes(_,len(A.shape)-2,-2)

	return out


@jax.jit
def flatten_first(X):
	blocksize=X.shape[0]*X.shape[1]
	shape=X.shape[2:]
	return jnp.reshape(X,(blocksize,)+shape)
	





@jax.jit
def allmatrixproducts(As,Bs):
	products=apply_on_n(As,Bs)
	return flatten_first(products)





def normalize(f,X_):

	scalesquared=sqloss(f(X_),0)
	C=1/math.sqrt(scalesquared)

	@jax.jit
	def g(X):
		return C*f(X)

	return g
