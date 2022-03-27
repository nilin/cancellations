import numpy as np
import math
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import util
import sys
import os
import shutil
import multiprocessing as mp
import permutations





def compute_unsigned_term(W,X,activation,p):
	I,n,d=W.shape
	S=X.shape[0]

	P=permutations.perm_as_matrix(p)
	pX=jax.vmap(jnp.dot,in_axes=(None,0))(P,X)

	W_=jnp.reshape(W,(I,n*d))	
	pX_=jnp.reshape(pX,(S,n*d))
	return activation(jnp.inner(W_,pX_))

@jax.jit
def compute_unsigned_term_ReLU(W,X,p):
	return compute_unsigned_term(W,X,util.ReLU,p)

@jax.jit
def compute_unsigned_term_HS(W,X,p):
	return compute_unsigned_term(W,X,util.heaviside,p)

@jax.jit
def compute_unsigned_term_tanh(W,X,p):
	return compute_unsigned_term(W,X,jnp.tanh,p)


@jax.jit
def zeros(W,X):
	instances,samples=W.shape[0],X.shape[0]
	return jnp.zeros(shape=(instances,samples))

def partial_sum(inputs):
	W,X,ac_name,start,smallblock=inputs
	compute_unsigned_term_activation=globals()['compute_unsigned_term_'+ac_name]
	S=zeros(W,X)
	n=W.shape[-2]
	p0=permutations.k_to_perm(start,n)
	p=p0
	sgn=permutations.sign(p0)
	for i in range(smallblock):
		S=S+sgn*compute_unsigned_term_activation(W,X,p)
		p,ds=permutations.nextperm(p)	
		sgn=sgn*ds
	return S


