# nilin

from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import math
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import util
from util import print_
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


@jax.jit
def apply_n(P,X):
	PX=jnp.dot(P,X)			# n,zip,s,d
	return jnp.moveaxis(PX,0,-2)	# zip,s,n,d

@jax.jit
def apply_many_to_n(Ps,X):		# Ps=p,n,n ; X=zip,s,n,d
	PX=jnp.dot(Ps,X)		# p,n,zip,s,d
	PX=jnp.moveaxis(PX,-2,0)	# s,p,n,zip,d
	return jnp.moveaxis(PX,-2,0)	# zip,s,p,n,d

@jax.jit
def contract(X,Y):
	return jnp.tensordot(X,Y,axes=([-1],[0]))


"""
Every permutation is of the form PQR where P<A,Q<B,R<C for some sets of permutations A,B,C, where, e.g., |A|=12*11*10, |B|=9*8, |C|=7!

PQRw*x=Rw*Q'P'x where (') denotes the inverse. There are |A| iterations of |B|*|C| permutations.
"""




@jax.jit
def GPU_batch_firstlayer(P,Q,RW,X):			# RW=zip,m,n,d ; X=zip,s,n,d
	Qt=jnp.swapaxes(Q,-2,-1)

	PtX=apply_n(P.T,X)				# zip,s,n,d
	QtPtX=apply_many_to_n(Qt,PtX)			# zip,s,q,n,d

	out=jax.vmap(dot_nd,in_axes=(0,0))(RW,QtPtX)	# zip,m,r,s,q
	return  jnp.moveaxis(out,-2,-3)			# zip,m,s,r,q	(previously m,s,r,q)



def sum_perms(W,X,permseqs,applylayers):		# W=zip,m,n,d ; X=zip,s,n,d

	try: lw,lx=len(W.shape),len(X.shape); assert lw==4 and lx==4 ;
	except AssertionError: util.print_('','Format should be W=zip,m,n,d; X=zip,s,n,d but they have ',lw,' and ',lx,' dimensions (4 and 4 required)');quit();

	(P,signP),(Q,signQ),(R,signR)=permseqs
	RW=apply_many_to_n(R,W)				#zip,m,r,n,d
	S=0
	for i in range(P.shape[0]):
		if P.shape[0]>1:
			print('  '+str(i)+'th/'+str(P.shape[0])+' block of '+str(permseqs[1][1].size)+'x'+str(permseqs[2][1].size)+' permutations done',end='\r')
		firstlayer=GPU_batch_firstlayer(P[i],Q,RW,X)
		permbatch=applylayers(firstlayer)
		summedpermbatch=jnp.dot(jnp.dot(permbatch,signQ),signR)

		S=S+signP[i]*summedpermbatch
	return S 					#zip,m,s


def gen_applylayers(Ws,ac_name):
	activation=util.activations[ac_name] #{'ReLU':util.ReLU,'tanh':jnp.tanh,'HS':util.heaviside,'DReLU':util.DReLU,'exp':jnp.exp}[ac_name]

	@jax.jit	
	def applylayers(X):
		for W in Ws:
			X=activation(X)
			X=jax.vmap(contract,in_axes=(0,0))(W,X)	
		return X
	return applylayers


def sum_perms_multilayer(Ws:list,Xs_,ac_name,mode='standard'):

	W=Ws[0]
	z,m,n,d=W.shape

	kQ,kR=blocksizechoices(n,mode)
	permseqs=perms.gen_complementary_Perm_seqs([n,kQ,kR])

	t=time.perf_counter()
	outputs=[]

	print_(mode,'n='+str(n)+'\n'+str(len(Ws))+' layers, '+ac_name+' activation')

	for s,Xs in enumerate(Xs_):


		outputs.append(sum_perms(W,Xs,permseqs,gen_applylayers(Ws[1:],ac_name)))
		t=printinfo(t,n,s,Xs.shape[0],len(Xs_))

	return jnp.concatenate(outputs,axis=-1)/jnp.sqrt(math.factorial(n))



def sum_perms_multilayer_zip(Ws:list,Xs:list,ac_name,mode='standard'):

	z,m,n,d=Ws[0][0].shape

	kQ,kR=blocksizechoices(n,mode)
	permseqs=perms.gen_complementary_Perm_seqs([n,kQ,kR])

	t=time.perf_counter()
	outputs=[]

	print_(mode,'n='+str(n)+'\n'+str(len(Ws))+' layers, '+ac_name+' activation')

	for s,_ in enumerate(Xs):

		outputs.append(sum_perms(Ws[s][0],Xs[s],permseqs,gen_applylayers(Ws[s][1:],ac_name)))
		t=printinfo(t,n,s,Xs[s].shape[0],len(Xs))

	out=jnp.concatenate(outputs,axis=0)/jnp.sqrt(math.factorial(n))

	return jnp.squeeze(out)


def blocksizechoices(n,mode):

	kQ=min(10,n)
	
	sidelengths=jnp.array([max(math.factorial(r),math.factorial(kQ)/math.factorial(r)) for r in range(1,kQ+1)])
	kR=jnp.argmin(sidelengths)+1

	if mode=='test':
		#make sure all 3 blocks are nontrivial
		kQ=n-1
		kR=kQ-1

	return kQ,kR







# print #####################################################################################################

def printpermseqs(permseqs,mode):
	print_(mode,str(permseqs[2][1].size)+'x'+str(permseqs[1][1].size)+' blocks of permutations x '+str(permseqs[0][1].size)+' iterations')	

def printinfo(t0,n,s,batchsize,batches,**kwargs):
	t1=time.perf_counter()
	dt=t1-t0
	if batchsize==1 and batches==1:
		sampleinfo=''
	else:
		sampleinfo=('batch '+str(s+1)+'/'+str(batches)+' containing '+str(batchsize)+' samples.' if batchsize>1 else 'sample '+str(s+1)+'/'+str(batches))
	bk.log('Permutations/time = '+'{:,}'.format(int(math.factorial(n)//dt))+'/second. '+sampleinfo,' (n=',n,')')
	if s==batches-1:
		print('')

	return t1




##def samplesizechoices(n,m):
##	singlesampleoutputsize=math.factorial(n)*m
##	return max(10**7//singlesampleoutputsize,1)
	



	
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
