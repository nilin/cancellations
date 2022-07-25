import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import pdb
from jax.lax import collapse	

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



@jax.jit
def collapselast(A,k):
	dims=A.shape
	#return collapse(A,dims-k,dims)
	return jnp.reshape(A,dims[:-2]+(dims[-2]*dims[-1],))


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




def chop(Xs,chunksize):
	S=Xs[0].shape[0]
	limits=[(a,min(a+chunksize,S)) for a in range(0,S,chunksize)]
	return [tuple([X[a:b] for X in Xs]) for a,b in limits]
	




def addgrads(G1,G2):
	if G1==None:
		return G2
	elif type(G2)==list:
		return [addgrads(g1,g2) for g1,g2 in zip(G1,G2)]
	else:
		return G1+G2
		
def scalegrad(G,r):
	if type(G)==list:
		return [scalegrad(g,r) for g in G]
	else:
		return r*G


def avg_grads(Gs):
	Gsum=None
	for G in Gs:
		Gsum=addgrads(Gsum,S)
	return scalegrad(Gsum,1/len(Gs))



def distinguishable(x,y,p_val=.10,**kwargs): # alternative='greater' to stop when no longer decreasing
	u,p=st.mannwhitneyu(x,y,**kwargs)
	return p<p_val



def donothing(*args):
	pass



def noparams(f_):
	@jax.jit
	def f(x):
		return f_(None,x)
	return f

def dummyparams(f):
	@jax.jit
	def f_(_,x):
		return f(x)
	return f_



