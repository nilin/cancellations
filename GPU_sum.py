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
import permutations as perms
import partialsum
import tqdm



@jax.jit
def GPU_batch_ReLU(P,Q,Rw,x,signQ,signR):
	return GPU_batch(P,Q,Rw,x,signQ,signR,util.ReLU)

#
#@jax.jit
#def sum_permblocks_term_HS(P,Qdata,Rdata):
#	return sum_permblocks(P,Qdata,Rdata,util.heaviside)
#
#@jax.jit
#def sum_permblocks_term_tanh(P,Qdata,Rdata):
#	return sum_permblocks(P,Qdata,Rdata,jnp.tanh)
#
#
#



def GPU_batch(P,Q,Rw,x,signQ,signR,activation):

	Ptx=jnp.dot(P.T,x)
	QtPtx=jnp.dot(jnp.swapaxes(Q,-2,-1),Ptx)

	ac_inputs=jnp.tensordot(Rw,QtPtx,axes=([-2,-1],[-2,-1]))
	ac_outputs=activation(ac_inputs)

	return jnp.inner(signR,jnp.inner(ac_outputs,signQ))




def sum_permblocks(w,x,ac_name,kR,kQ,start,stop):
	n=w.shape[-2]
	(q,signQ),(r,signR)=perms.generate_complementary_perm_seqs([kR,kQ],n=n)
	Q=perms.perm_as_matrix(q)
	R=perms.perm_as_matrix(r)
	Rw=jnp.dot(R,w)
	
	GPU_batch_func=globals()['GPU_batch_'+ac_name]

	blocksize=math.factorial(kQ)
	assert(start%blocksize==0 and stop%blocksize==0)	
	K0=start

	p=perms.k_to_perm(K0,n)
	signP=perms.sign(p)

	S=0
	#for K in tqdm.tqdm(range(start,stop,blocksize)):
	for K in range(start,stop,blocksize):
		S=S+signP*GPU_batch_func(perms.perm_as_matrix(p),Q,Rw,x,signQ,signR)
		p,ds=perms.nextblock(p,kQ)
		signP=signP*ds
	return S


def sum_all_perms(w,x,ac_name):
	n=w.shape[-2]
	N=math.factorial(n)
	kR,kQ=blocksizechoices(n)
	return sum_permblocks(w,x,ac_name,kR,kQ,0,N)/jnp.sqrt(1.0*N)



def blocksizechoices(n):
	kQ=min(11,n-1)
	kR=min(7,kQ-1)
	return kR,kQ
	

def test(n,d):
	key=jax.random.PRNGKey(0)
	key1,key2=jax.random.split(key)
	w=jax.random.normal(key1,(n,d))/jnp.sqrt(n*d)
	x=jax.random.normal(key2,(n,d))

	print(util.L2norm(w))

	S=sum_all_perms(w,x,'ReLU')
	print(S)



