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




def compute_unsigned_orbit(W,X,activation,p,orbit,orbitsigns):
	I,n,d=W.shape
	S=X.shape[0]

	P=permutations.perm_as_matrix(p)
	pX=jax.vmap(jnp.dot,in_axes=(None,0))(P,X)
	qpX=jax.vmap(jnp.dot,in_axes=(0,None))(orbits,pX)

	W_=jnp.reshape(W,(I,n*d))	
	qpX_=jnp.reshape(qpX,(S,n*d))
	return jnp.inner(orbitsigns,activation(jnp.inner(W_,qpX_)))

@jax.jit
def compute_unsigned_term_ReLU(W,X,p):
	return compute_unsigned_term(W,X,util.ReLU,p)

@jax.jit
def compute_unsigned_term_HS(W,X,p):
	return compute_unsigned_term(W,X,util.heaviside,p)

@jax.jit
def compute_unsigned_term_tanh(W,X,p):
	return compute_unsigned_term(W,X,jnp.tanh,p)


def partial_sum_gpu(W,X,ac_name,start,smallblock,orbit,m):
	n=W.shape[-2]
	k=n-m	
	
	compute_unsigned_term_activation=globals()['compute_unsigned_term_'+ac_name]
	S=jnp.zeros(W.shape[0],X.shape[0],k)
	p0=permutations.k_to_perm(start,n)
	p=p0
	sgn=permutations.sign(p0)
	for i in range(smallblock):
		S=S+sgn*compute_unsigned_orbit_activation(W,X,list(range(k))+[pj+k or pj in p],orbit)
		p,ds=permutations.nextperm(p)	
		sgn=sgn*ds
	return S



