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



@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))




"""
Every permutation is of the form PQR where P<A,Q<B,R<C for some sets of permutations A,B,C, where, e.g., |A|=12*11*10, |B|=9*8, |C|=7!

PQRw*x=Rw*Q'P'x where (') denotes the inverse. There are |A| iterations of |B|*|C| permutations.
"""




@jax.jit
def GPU_batch_firstlayer(P,Q,RW,X):

	PtX=jax.vmap(jnp.dot,in_axes=(None,0))(P.T,X)
	QtPtX=jax.vmap(jnp.dot,in_axes=(None,0))(jnp.swapaxes(Q,-2,-1),PtX)

	return jnp.moveaxis(dot_nd(RW,QtPtX),-2,-3)



def sum_perms(W,X,permseqs,applylayers):
	(P,signP),(Q,signQ),(R,signR)=permseqs
	RW=jax.vmap(jnp.dot,in_axes=(None,0))(R,W)
	S=0
	for i in range(P.shape[0]):
		if P.shape[0]>1:
			if i==0:
				print('\n'+100*'-')
			print('  '+str(i)+'th/'+str(P.shape[0])+' block of '+str(permseqs[1][1].size)+'x'+str(permseqs[2][1].size)+' permutations done',end='\r')
		firstlayer=GPU_batch_firstlayer(P[i],Q,RW,X)
		permbatch=applylayers(firstlayer)
		summedpermbatch=jnp.inner(signR,jnp.dot(permbatch,signQ))

		S=S+signP[i]*summedpermbatch
	return S #jnp.swapaxes(S,0,1)


def gen_applylayers(Ws,ac_name):
	activation={'ReLU':util.ReLU,'tanh':jnp.tanh,'HS':util.heaviside,'DReLU':util.DReLU}[ac_name]
	def applylayers(X):
		for W in Ws:
			X=activation(X)
			X=jnp.tensordot(W,X,axes=([-1],[0]))
		return X
	return applylayers


def sum_perms_multilayer(Ws:list,Xs_,ac_name,mode='standard'):

	W=Ws[0]
	m,n,d=W.shape
	print('n='+str(n)+'\n'+str(len(Ws))+' layers')

	kQ,kR=blocksizechoices(n,mode)
	permseqs=perms.gen_complementary_Perm_seqs([n,kQ,kR])
	print(str(permseqs[2][1].size)+'x'+str(permseqs[1][1].size)+' blocks of permutations x '+str(permseqs[0][1].size)+' iterations')

	t=time.perf_counter()
	sum_=0

	for s,Xs in enumerate(Xs_):
		sum_=sum_+sum_perms(W,Xs,permseqs,gen_applylayers(Ws[1:],ac_name))
		t=printinfo(t,n,s,Xs.shape[0],len(Xs_))

	print('\n')

	return sum_/jnp.sqrt(math.factorial(n))


def printinfo(t0,n,s,batchsize,batches):
	t1=time.perf_counter()
	dt=t1-t0
	timesbatchsize=(' x '+str(batchsize)+' samples (batch ' if batchsize>1 else ' (sample ')
	print('Permutations/time = '+'{:,}'.format(int(math.factorial(n)//dt))+'/second.'+timesbatchsize+str(s+1)+'/'+str(batches)+')',end='\r')
	return t1

	
#def sum_perms_multilayer(Ws:list,X_,ac_name):
#
#	print(str(len(Ws))+' layers')
#
#	W=Ws[0]
#	m,n,d=W.shape
#
#	kQ,kR=blocksizechoices(n)
#	permseqs=perms.gen_complementary_Perm_seqs([n,kQ,kR])
#	print(str(permseqs[1][1].size)+'x'+str(permseqs[2][1].size)+' blocks of permutations')
#
#	outputs=[]
#	t0=time.perf_counter()
#
#	for i in range(0,X_.shape[0]):
#
#		x=X_[i]
#		x_=jnp.repeat(jnp.expand_dims(x,axis=0),m,axis=0)
#	
#		outputs.append(jnp.squeeze(sum_perms(W,x_,permseqs,gen_applylayers(Ws[1:],ac_name))))
#
#		t1=time.perf_counter()
#	
#		print('Permutations/time = '+'{:,}'.format(int(math.factorial(n)//(t1-t0)))+'/second. Samples done:'+str(i+1)+'/'+str(X_.shape[0]),end='\r')
#		t0=t1
#
#	print('\n')
#
#	return jnp.array(outputs)
	
	

def blocksizechoices(n,mode):

	kQ=min(10,n)
	#kR=max(1,min(6,kQ-3))
	
	sidelengths=jnp.array([max(math.factorial(r),math.factorial(kQ)/math.factorial(r)) for r in range(1,kQ+1)])
	kR=jnp.argmin(sidelengths)+1

	if mode=='test':
		#make sure all 3 blocks are nontrivial
		kQ=n-1
		kR=kQ-1

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
