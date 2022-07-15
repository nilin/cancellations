import GPU_sum
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import GPU_sum
import optax
import math
import universality
import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools




				
		
def error(X,Y):
	return universality.sqloss(X,Y)/universality.sqloss(0,Y)

def testansatz(histfile,samples=10):
	W,b=bk.get(histfile)[-1]
	m,n,d=W[0].shape
	X=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
	X=X[:samples]

	comp_NS_AS(W,b,X)
	

def comp_NS_AS(W,b,X):
	out=universality.sumperms(W,b,X)

	f=lambda X:universality.nonsym(W,b,X)
	true=naiveAS(f,X)

	assert error(out,true)<.01


def naiveAS(f,X):
	n,d=X.shape[-2:]
	I=jnp.eye(n)

	out=0
	for p in itertools.permutations(I):
		P=jnp.array(p)
		sign=jnp.linalg.det(P)
		PX=jnp.dot(P,X)
		out=out+sign*f(PX)			

	return out
	



if __name__=="__main__":

	d=3
	n=4
	histfile='data/hists/AS_'+bk.formatvars_({'d':d,'n':n,'m':100})
	testansatz(histfile)

	
