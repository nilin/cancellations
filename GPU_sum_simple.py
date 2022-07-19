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
import permutations_simple as ps
import typing
import testing
import pdb



@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))




def AS_func(W0,func,X):
	n=W0.shape[-2]					# W0:	m,n,d
	Ps,signs=ps.allperms(n)			# Ps:	n!,n,n
	PW0=util.apply_on_n(Ps,W0)			# PW0:	n!,m,n,d
	L1=dot_nd(PW0,X)				# L1:	n!,m,s	
	Y_NS=func(L1)					# Y_NS:	n!,s
	return jnp.dot(signs,Y_NS)			# s



def NS_func(W0,func,X):
	L1=dot_nd(W0,X)					# L1:	m,s
	return func(L1)					# s


def AS_NN(Ws,bs,X,ac='ReLU'):
	applylayers=gen_applylayers(Ws[1:],bs,ac)
	return AS_func(Ws[0],applylayers,X)

def NN(W,b,X,ac='ReLU'):
	applylayers=gen_applylayers(Ws[1:],bs,ac)
	return NS_func(Ws[0],applylayers,X)



def gen_applylayers(Ws,bs,ac_name):
	activation=util.activations[ac_name]

	@jax.jit	
	def applylayers(Y):
		for W,b in zip(Ws,bs):
			Yb=jax.vmap(jnp.add,in_axes=(-2,0),out_axes=-2)(Y,b)
			acYb=activation(Yb)
#			pdb.set_trace()
			Y=util.apply_on_n(W,acYb)	
		return jnp.squeeze(Y)
	return applylayers




#---------------------------------------------------------------------------------------------------- 
# test
#---------------------------------------------------------------------------------------------------- 


def test_AS(Ws,bs,X):

	Af=lambda x:AS_NN(Ws,bs,x)
	f=lambda x:NN(Ws,bs,x)

	testing.test_AS(Af,f,X)


	

