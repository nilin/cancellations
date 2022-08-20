#
# nilin
#
# 2022/7
#




#----------------------------------------------------------------------------------------------------
# This file replaces GPU_sum
#----------------------------------------------------------------------------------------------------


import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import util
import sys
import os
import shutil
import permutations_simple as ps
import typing
import testing
import config as cfg
from jax.lax import collapse
import pdb
import AS_HEAVY
import multivariate as mv
import backflow as bf




#=======================================================================================================
#
# explicit AS
#
# basic AS (n=1,..,8)
#
#=======================================================================================================

def gen_Af_simple(n,f):
	Ps,signs=ps.allperms(n)					# Ps:	n!,n,n

	@jax.jit
	def Af(params,X):
		PX=util.apply_on_n(Ps,X)				# PX:	n!,s,n,d
		fX=f(params,PX)						# fX:	n!,s
		return jnp.dot(signs,fX)				# s

	return Af


	



# combine light and heavy regimes 
#----------------------------------------------------------------------------------------------------

def gen_Af(n,f):
	return gen_Af_simple(n,f) if n<=cfg.heavy_threshold else gen_Af_heavy(n,f)

def gen_lossgrad_Af(n,f,lossfn):
	return mv.gen_lossgrad(gen_Af(n,f)) if n<=cfg.heavy_threshold else AS_HEAVY.gen_lossgrad_Af_heavy(n,f,lossfn)

		




#=======================================================================================================
#
# backflow+det
#
#=======================================================================================================



@jax.jit
def detsum(A,Y):
	snkn=jnp.inner(Y,A)
	sknn=jnp.swapaxes(snkn,-3,-2)
	return jnp.sum(jnp.linalg.det(sknn),axis=-1)

def gen_backflow0(activation):
	return util.compose(bf.gen_backflow(activation),detsum)

#=======================================================================================================
	
@jax.jit
def EV_to_sym(b,Y):
	return jnp.inner(jnp.sum(Y,axis=-2),b)


@jax.jit
def EV_to_antisym(Ab,Y):
	A,b=Ab
	return detsum(A,Y)*EV_to_sym(b,Y)


def gen_backflow1(activation):
	return util.compose(bf.gen_backflow(activation),EV_to_antisym)






#=======================================================================================================

#===================
# Example: ferminet
#===================
#
#def gen_ferminet(n,ac='tanh'):
#	return util.recompose(bf.gen_FN_backflow(ac),get_detsum(n))
#




#=======================================================================================================
# Slater
#=======================================================================================================

"""
m*n separate functions with distinct weights
"""
def gen_Slater(n,phi):

	@jax.jit
	def Af(weights,X):
		matrices=jnp.stack([jnp.stack([phi(weights[i],X[:,j,:]) for j in range(n)],axis=-1) for i in range(n)],axis=-1)
		return jnp.linalg.det(matrices)
	return Af


#"""
## F:x->(f1(x),..,fn(x))		s,d |-> s,n
#"""
#def vectorSlater(fs):								
#	Fs=jax.vmap(fs,in_axes=(None,1),out_axes=-1)
#
#	@jax.jit
#	def AF(params,X):
#		FX=Fs(params,X)			# FX:	s,n (basisfunction),n (particle)
#		return jnp.linalg.det(FX)
#	return AF

#=======================================================================================================
## test
#=======================================================================================================

def test_AS(Ws,bs,X):

	Af=lambda x:AS_NN(Ws,bs,x)
	f=lambda x:NN(Ws,bs,x)

	testing.test_AS(Af,f,X)


if __name__=='__main__':
	pass	
