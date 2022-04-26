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
import typing
import testing


#
#@jax.jit
#def GPU_batch_ReLU(P,Q,Rw,x):
#	return GPU_batch_firstlayer(P,Q,Rw,x,util.ReLU)
#@jax.jit
#def GPU_batch_HS(P,Q,Rw,x):
#	return GPU_batch_firstlayer(P,Q,Rw,x,util.heaviside)
#@jax.jit
#def GPU_batch_tanh(P,Q,Rw,x):
#	return GPU_batch_firstlayer(P,Q,Rw,x,jnp.tanh)
#


@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))


@jax.jit
def GPU_batch_firstlayer(P,Q,RW,X):

	PtX=jax.vmap(jnp.dot,in_axes=(None,0))(P.T,X)
	QtPtX=jax.vmap(jnp.dot,in_axes=(None,0))(jnp.swapaxes(Q,-2,-1),PtX)

	return jax.vmap(dot_nd,in_axes=(0,0))(RW,QtPtX)



def sum_perms(W,X,permseqs,applylayers):
	(P,signP),(Q,signQ),(R,signR)=permseqs
	RW=jax.vmap(jnp.dot,in_axes=(None,0))(R,W)
	S=0
	for i in range(P.shape[0]):
		firstlayer=GPU_batch_firstlayer(P[i],Q,RW,X)
		permbatch=applylayers(firstlayer)
		summedpermbatch=jnp.inner(signR,jnp.dot(permbatch,signQ))

		S=S+signP[i]*summedpermbatch
	return S


def gen_applylayers(Ws,ac_name):
	activation={'ReLU':util.ReLU,'tanh':jnp.tanh,'HS':util.heaviside,'DReLU':util.DReLU}[ac_name]
	def applylayers(X):
		for W in Ws:
			X=activation(X)
			X=jnp.tensordot(W,X,axes=([-1],[0]))
		return X
	return applylayers


def sum_perms_multilayer(Ws:list,X_,ac_name):
	W=Ws[0]
	m,n,d=W.shape

	kQ,kR=blocksizechoices(n)
	permseqs=perms.gen_complementary_Perm_seqs([n,kQ,kR])

	outputs=[]
	t0=time.perf_counter()

	for i in range(0,X_.shape[0]):

		x=X_[i]
		x_=jnp.repeat(jnp.expand_dims(x,axis=0),m,axis=0)
	
		outputs.append(jnp.squeeze(sum_perms(W,x_,permseqs,gen_applylayers(Ws[1:],ac_name))))

		t1=time.perf_counter()
		print('Permutations/time = '+'{:,}'.format(int(math.factorial(n)//(t1-t0)))+'/second. Samples done:'+str(i)+'/'+str(X_.shape[0]),end='\r')
		t0=t1

	print('\n')

	return jnp.array(outputs)
	
	

def blocksizechoices(n):
	kQ=min(10,n)
	kR=max(1,min(6,kQ-3))
	return kQ,kR


def samplesizechoices(n,m):
	singlesampleoutputsize=math.factorial(n)*m
	return max(10**7//singlesampleoutputsize,1)
	

#def sum_perms_(w,x,ac_name):	
#
#	n=w.shape[-2]
#	
#	permbatchsize=math.factorial(kQ)
#	instancebatchsize=4*(10**8)//permbatchsize
#	start=0
#	outputs=[]
#	for start in range(0,W.shape[0],instancebatchsize):
#		end=min(start+instancebatchsize,W.shape[0])
#		W_batch=W[start:end]
#		X_batch=X[start:end]
#
#		t0=time.perf_counter()
#		outputs.append(sum_perms_instancebatch(W_batch,X_batch,permseqs,GPU_batch_func))
#		t1=time.perf_counter()
#		bk.log('sample batch '+str(start)+'-'+str(end)+30*' '+'('+str((1.0*math.factorial(n)/(t1-t0))//(10**6))+' million permutations)x('+str(end-start)+' instances) per second')
#	return jnp.concatenate(outputs,axis=0)




	
################################################# test ###################################################



	
		


#	print(testing.naive_sum_test(Ws,X,ac='tanh'))
#	t2=time.perf_counter()
#	print('time for naive algorithm: '+str(t2-t1))



#
#def test():
#	d=3;n=5
#	key=jax.random.PRNGKey(0)
#	key1,key2,key3,key4,*r=jax.random.split(key,5)
#	W=jax.random.normal(key1,(10,n,d))/jnp.sqrt(n*d)
#	X=jax.random.normal(key2,(10,n,d))
#
#	test_batch(W,X)
#	test_det(W,X)
#	test_small_n_edgecase(W,X)
#	
#	n=12
#	W=jax.random.normal(key3,(100,n,d))/jnp.sqrt(n*d)
#	X=jax.random.normal(key4,(100,n,d))
#	
#	_=input('press enter for speed test ')
#	speedtest(W,X)	
#
#def test_small_n_edgecase(W_,X_):
#	W=jnp.take(W_,jnp.arange(2),axis=-2)
#	X=jnp.take(X_,jnp.arange(2),axis=-2)
#	S=[jnp.tanh(jnp.vdot(W[i],X[i]))-jnp.tanh(jnp.vdot(jnp.flip(W[i],axis=-2),X[i])) for i in range(W.shape[0])]
#	util.assertequal(S,sum_perms(W,X,'tanh'),'n=2 edge case')
#
#def test_batch(W,X):
#	n=W.shape[-2]
#	samples=W.shape[0]
#	out=[]
#	for s in range(samples):
#		w,x=W[s],X[s]
#		S=0
#		for i in range(math.factorial(n)):
#			P=perms.k_to_matrix(i,n)
#			sgn=jnp.linalg.det(P)
#			S=S+sgn*util.ReLU(jnp.tensordot(jnp.dot(P,w),x))
#		out.append(S)
#	util.assertequal(sum_perms(W,X,'ReLU'),jnp.array(out),'sum_all_perms')
#
#def test_det(W,X):
#	dets=[jnp.linalg.det(jnp.exp(jnp.inner(W[i],X[i]))) for i in range(W.shape[0])]
#	util.assertequal(sum_perms(W,X,'exp'),dets,'exp Slater')
#
#def speedtest(w,x):
#	sum_perms(w,x,'ReLU')



if __name__=='__main__':
	if len(sys.argv)>1 and sys.argv[1]=='t':
		testing.test_multilayer(n=int(input('n: ')),layers=int(input('layers: ')))
