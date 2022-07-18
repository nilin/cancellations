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



@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))




def AS(W0,applylayers,X):
	n=W0.shape[-2]					# W0:	m,n,d
	Ps,signs=permutations.allperms(n)		# Ps:	n!,n,n
	PW0=util.apply_on_n(Ps,W0)			# PW0:	n!,m,n,d
	L1=dot_nd(PW0,X)				# L1:	n!,m,s	
	Y_NS=applylayers(L1)				# Y_NS:	n!,s
	return jnp.dot(self.signs,Y_NS)			# s



def NS(W0,applylayers,X):
	L1=dot_nd(W0,X)					# L1:	m,s
	return applylayers(L1)				# s


def AS_NN(Ws,bs,X,ac='ReLU'):
	applylayers=gen_applylayers(Ws[1:],bs,ac)
	return AS(Ws[0],applylayers,X)

def NN(W,b,X,ac='ReLU'):
	applylayers=gen_applylayers(Ws[1:],bs,ac)
	return NS(Ws[0],applylayers,X)



def gen_applylayers(Ws,bs,ac_name):
	activation=util.activations[ac_name]

	@jax.jit	
	def applylayers(Y):
		for W,b in zip(Ws,bs):
			Yb=jax.vmap(jnp.add,in_axes=(-2,0))(Y,b)
			acYb=activation(Yb)
			Y=util.apply_on_n(W,Y)	
		return jnp.squeeze(Y)
	return applylayers




#---------------------------------------------------------------------------------------------------- 
# test
#---------------------------------------------------------------------------------------------------- 


