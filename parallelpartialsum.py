import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
import seaborn as sns
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import optax
import util
import sys
import os
import shutil
import cancellation as canc
import proxies
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

def single_process_partial_sum(inputs):
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


def parallel_sum(W,X,ac_name,start,stop,prefix,params):

	tasks, smallblock, bigblock=params['tasks'],params['smallblock'],params['bigblock']

	assert(5040%bigblock==0)

	a=start
	cumulative_sum=zeros(W,X)
	timer=bk.Stopwatch()
	n=W.shape[-2]
	N=math.factorial(n)
	prevfilepath='nonexistent'

	with mp.Pool(tasks) as pool:
		while a<stop:
			inputs=[a+smallblock*t for t in range(tasks)]
			inputs_=[(W,X,ac_name,k,smallblock) for k in inputs]
			parallelsmallsums=pool.map(single_process_partial_sum,inputs_)
			cumulative_sum=cumulative_sum+sum(parallelsmallsums)
			a=a+bigblock
		
			filepath=prefix+str(start)+' '+str(a)			
			bk.savedata({'result':cumulative_sum,'interval':(start,a),'W':W,'X':X},filepath)
	
			if os.path.exists('data/'+prevfilepath): 
				removepath='data/'+prevfilepath
				os.remove(removepath)
			prevfilepath=filepath
			bk.printbar(a/N,str(a)+' terms. '+str(round(bigblock/timer.tick()))+' terms per second.')

	print('Reached '+str(stop))



"""
gen_partial_sum.py ReLU n 10
"""



if __name__=='__main__':
	ac_name=sys.argv[1]
	Wtype=canc.Wtypes[sys.argv[2]]
	n=int(sys.argv[3])
	start=int(input('At which k to start the partial sum? '))
	stop=math.factorial(n)

	dirpath='partialsums/'+Wtype
	bk.mkdir('data/'+dirpath)

	W,X=[bk.getdata(Wtype+'/WX')[k][n] for k in ('Ws','Xs')]
	
	print('Computing partial sum for '+ac_name+' activation, '+Wtype+' weights, and n='+str(n)+'.')
	prefix=dirpath+'/'+ac_name+' n='+str(n)+' range='

	
	parallel_sum(W,X,ac_name,start,stop,prefix,{'tasks':8,'smallblock':630,'bigblock':5040})
