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
@jax.jit
def GPU_batch_HS(P,Q,Rw,x,signQ,signR):
	return GPU_batch(P,Q,Rw,x,signQ,signR,util.heaviside)
@jax.jit
def GPU_batch_tanh(P,Q,Rw,x,signQ,signR):
	return GPU_batch(P,Q,Rw,x,signQ,signR,jnp.tanh)
@jax.jit
def GPU_batch_exp(P,Q,Rw,x,signQ,signR):
	return GPU_batch(P,Q,Rw,x,signQ,signR,jnp.exp) #for testing






def GPU_batch(P,Q,Rw,x,signQ,signR,activation):

	Ptx=jnp.dot(P.T,x)
	QtPtx=jnp.dot(jnp.swapaxes(Q,-2,-1),Ptx)

	ac_inputs=jnp.tensordot(Rw,QtPtx,axes=([-2,-1],[-2,-1]))
	ac_outputs=activation(ac_inputs)

	return jnp.inner(signR,jnp.inner(ac_outputs,signQ))




def sum_permblocks(w,x,ac_name,kR,kQ,start,stop,loud=False):
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
	for K in range(start,stop,blocksize):
		S=S+signP*GPU_batch_func(perms.perm_as_matrix(p),Q,Rw,x,signQ,signR)
		p,ds=perms.nextblock(p,kQ)
		signP=signP*ds
		
		if loud:
			print('Q size '+str(Q.shape)+'  |  '+'Rw size '+str(Rw.shape)+'  |  permutation number '+str(K))

	return S


def sum_all_perms(w,x,ac_name,**kwargs):
	n=w.shape[-2]
	N=math.factorial(n)
	kR,kQ=blocksizechoices(n)
	return sum_permblocks(w,x,ac_name,kR,kQ,0,N,**kwargs)



def blocksizechoices(n):
	kQ=min(12,n-1)
	kR=min(8,kQ-1)
	return kR,kQ
	


	
################################################# test ###################################################

def test():
	d=3;n=5
	key=jax.random.PRNGKey(0)
	key1,key2,key3,key4,*r=jax.random.split(key,5)
	w=jax.random.normal(key1,(n,d))/jnp.sqrt(n*d)
	x=jax.random.normal(key2,(n,d))

	test_batch(w,x)
	test_det(w,x)
	
	n=13
	w=jax.random.normal(key3,(n,d))/jnp.sqrt(n*d)
	x=jax.random.normal(key4,(n,d))

	speedtest(w,x)	

def test_batch(w,x):
	n=w.shape[0]
	S1=0
	for i in range(12):
		P=perms.k_to_matrix(i,n)
		sgn=jnp.linalg.det(P)
		S1=S1+sgn*util.ReLU(jnp.tensordot(jnp.dot(P,w),x,axes=([0,1],[0,1])))
	S2=sum_permblocks(w,x,'ReLU',2,3,0,12)
	util.assertequal(S1,S2)
	S3=0
	for i in range(math.factorial(n)):
		P=perms.k_to_matrix(i,n)
		sgn=jnp.linalg.det(P)
		S3=S3+sgn*util.ReLU(jnp.tensordot(jnp.dot(P,w),x,axes=([0,1],[0,1])))
	S4=sum_permblocks(w,x,'ReLU',2,3,0,math.factorial(n))
	S5=sum_all_perms(w,x,'ReLU')
	util.assertequal(S3,S4)
	util.assertequal(S3,S5)

def test_det(w,x):
	util.assertequal(sum_all_perms(w,x,'exp'),jnp.linalg.det(jnp.exp(jnp.inner(w,x))))

def speedtest(w,x):
	sum_all_perms(w,x,'ReLU',loud=True)



if __name__=='__main__':
	if len(sys.argv)>1 and sys.argv[1]=='t':
		test()
