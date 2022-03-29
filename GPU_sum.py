# nilin

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



@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))


def GPU_batch(P,Q,RW,X,signQ,signR,activation):

	PtX=jax.vmap(jnp.dot,in_axes=(None,0))(P.T,X)
	QtPtX=jax.vmap(jnp.dot,in_axes=(None,0))(jnp.swapaxes(Q,-2,-1),PtX)

	ac_inputs=jax.vmap(dot_nd,in_axes=(0,0))(RW,QtPtX)
	ac_outputs=activation(ac_inputs)

	return jnp.inner(signR,jnp.inner(ac_outputs,signQ))


def sum_perms(W,X,ac_name):	

	if(len(W.shape)==2):
		W=jnp.expand_dims(W,axis=0)
		X=jnp.expand_dims(X,axis=0)

	n=W.shape[-2]

	kQ,kR=blocksizechoices(n)
	permseqs=perms.gen_complementary_Perm_seqs([n,kQ,kR])
	
	GPU_batch_func=globals()['GPU_batch_'+ac_name]

	permbatchsize=math.factorial(kQ)
	instancebatchsize=4*(10**8)//permbatchsize
	start=0
	outputs=[]
	for start in range(0,W.shape[0],instancebatchsize):
		end=min(start+instancebatchsize,W.shape[0])
		bk.log('\nsample batch '+str(start)+'-'+str(end)+100*'-')
		W_batch=W[start:end]
		X_batch=X[start:end]
		outputs.append(sum_perms_instancebatch(W_batch,X_batch,permseqs,GPU_batch_func))
	return jnp.concatenate(outputs,axis=0)


def sum_perms_instancebatch(W,X,permseqs,GPU_batch_func):
	(P,signP),(Q,signQ),(R,signR)=permseqs
	RW=jax.vmap(jnp.dot,in_axes=(None,0))(R,W)
	S=0
	for i in range(P.shape[0]):
		S=S+signP[i]*GPU_batch_func(P[i],Q,RW,X,signQ,signR)
		bk.log('permutation number '+str(i*signQ.size*signR.size))
	return S


def blocksizechoices(n):
	kQ=min(11,n)
	kR=min(7,kQ-1)
	return kQ,kR
	


	
################################################# test ###################################################

def test():
	d=3;n=5
	key=jax.random.PRNGKey(0)
	key1,key2,key3,key4,*r=jax.random.split(key,5)
	W=jax.random.normal(key1,(10,n,d))/jnp.sqrt(n*d)
	X=jax.random.normal(key2,(10,n,d))

	test_batch(W,X)
	test_det(W,X)
	test_small_n_edgecase(W,X)
	
	n=12
	W=jax.random.normal(key3,(100,n,d))/jnp.sqrt(n*d)
	X=jax.random.normal(key4,(100,n,d))
	
	_=input('press enter for speed test ')
	speedtest(W,X)	

def test_small_n_edgecase(W_,X_):
	W=jnp.take(W_,jnp.arange(2),axis=-2)
	X=jnp.take(X_,jnp.arange(2),axis=-2)
	S=[jnp.tanh(jnp.vdot(W[i],X[i]))-jnp.tanh(jnp.vdot(jnp.flip(W[i],axis=-2),X[i])) for i in range(W.shape[0])]
	util.assertequal(S,sum_perms(W,X,'tanh'),'n=2 edge case')

def test_batch(W,X):
	n=W.shape[-2]
	samples=W.shape[0]
	out=[]
	for s in range(samples):
		w,x=W[s],X[s]
		S=0
		for i in range(math.factorial(n)):
			P=perms.k_to_matrix(i,n)
			sgn=jnp.linalg.det(P)
			S=S+sgn*util.ReLU(jnp.tensordot(jnp.dot(P,w),x))
		out.append(S)
	util.assertequal(sum_perms(W,X,'ReLU'),jnp.array(out),'sum_all_perms')

def test_det(W,X):
	dets=[jnp.linalg.det(jnp.exp(jnp.inner(W[i],X[i]))) for i in range(W.shape[0])]
	util.assertequal(sum_perms(W,X,'exp'),dets,'exp Slater')

def speedtest(w,x):
	sum_perms(w,x,'ReLU')



if __name__=='__main__':
	if len(sys.argv)>1 and sys.argv[1]=='t':
		test()
