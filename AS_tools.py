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
#from dets import DETS




#=======================================================================================================
# basic AS (n=1,..,8)
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
# special case: tensor product -> Slater
#=======================================================================================================

"""
# F:x->(f1(x),..,fn(x))		s,d |-> s,n
"""
def Slater(fs):								
	Fs=jax.vmap(fs,in_axes=(None,1),out_axes=-1)

	@jax.jit
	def AF(params,X):
		FX=Fs(params,X)			# FX:	s,n (basisfunction),n (particle)
		return jnp.linalg.det(FX)
	return AF





"""
m*n separate functions with distinct weights
"""
def gen_Slater(n,phi):

	@jax.jit
	def Af(weights,X):
		matrices=jnp.stack([jnp.stack([phi(weights[i],X[:,j,:]) for j in range(n)],axis=-1) for i in range(n)],axis=-1)
		return jnp.linalg.det(matrices)
	return Af


def gen_SlaterSum(n,phi):
	return mv.sum_f(gen_Slater(n,phi))	




"""
#
#def gen_SlaterSum(n,m,phi):
#	return jax.jit(mv.addf(*[phi]*m))
#
#
#def gen_SlaterSum_singlePhi(n,phi):
#
#	@jax.jit
#	def Af(weights,X):
#
#		s_mn_n=jnp.stack([phi(weights,X[:,j,:]) for j in range(n)],axis=-1)
#
#		s,mn,_=s_mn_n.shape; m=mn//n
#		smnn=jnp.reshape(s_mn_n,(s,m,n,n))
#
#		return jnp.sum(jnp.linalg.det(smnn),axis=1)
#
#	return Af
#
#
## phi_i(x) of k'th Slater = phi(weights[i],x)[k]
#
#def gen_SlaterSum_nPhis(n,phi):
#
#	@jax.jit
#	def Af(weights,X):
#
#		nnsm=jnp.array([[phi(weights[i],X[:,j,:]) for j in range(n)] for i in range(n)])
#		for _ in range(2):
#			nnsm=jnp.moveaxis(nnsm,0,-1)
#		return jnp.sum(jnp.linalg.det(nnsm),axis=1)
#
#	return Af
"""




			




#=======================================================================================================
## test
#=======================================================================================================

def test_AS(Ws,bs,X):

	Af=lambda x:AS_NN(Ws,bs,x)
	f=lambda x:NN(Ws,bs,x)

	testing.test_AS(Af,f,X)


if __name__=='__main__':
	pass	
