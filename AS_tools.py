# nilin

#----------------------------------------------------------------------------------------------------
# This file replaces GPU_sum
#----------------------------------------------------------------------------------------------------


from jax.config import config
#config.update("jax_enable_x64", True)
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
import permutations_simple as ps
import typing
import testing
from jax.lax import collapse
import pdb
from multivariate import NN_NS






#=======================================================================================================
# basic AS (n=1,..,8)
#=======================================================================================================

def gen_Af(n,f):
	Ps,signs=ps.allperms(n)					# Ps:	n!,n,n

	@jax.jit
	def Af(params,X):
		PX=util.apply_on_n(Ps,X)				# PX:	n!,s,n,d
		fX=f(params,PX)						# fX:	n!,s
		return jnp.dot(signs,fX)				# s

	return Af





def gen_lossgrad_Af(n,f,lossfn):
	Af=gen_Af(n,f)

	@jax.jit
	def collectiveloss(params,X,Y):
		fX=Af(params,X)
		return lossfn(fX,Y)

	@jax.jit	
	def lossgrad(params,X,Y):
		loss,grad=jax.value_and_grad(collectiveloss)(params,X,Y)
		return grad,loss

	return lossgrad






		
#=======================================================================================================
# combine light and heavy regimes 
#=======================================================================================================


from AS_HEAVY import gen_Af_heavy,gen_lossgrad_Af_heavy,heavy_threshold,emptytracker



def gen_AS_NN(n,tracker=emptytracker):
	return gen_Af(n,NN_NS) if n<=heavy_threshold else gen_Af_heavy(n,NN_NS,tracker=tracker)



def gen_lossgrad_AS_NN(n,lossfn,tracker=emptytracker):
	return gen_lossgrad_Af(n,NN_NS,lossfn) if n<=heavy_threshold else gen_lossgrad_Af_heavy(n,NN_NS,lossfn,tracker=tracker)
		








#=======================================================================================================
# special case: tensor product -> Slater
#=======================================================================================================



def Slater(F):								# F:x->(f1(x),..,fn(x))		s,d |-> s,n
	@jax.jit
	def AF(params,X):
		FX=jax.vmap(F,in_axes=(None,1),out_axes=-1)(params,X)	# FX:	s,n (basisfunction),n (particle)
		return jnp.linalg.det(FX)
	return AF









#=======================================================================================================
## test
#=======================================================================================================


def test_AS(Ws,bs,X):

	Af=lambda x:AS_NN(Ws,bs,x)
	f=lambda x:NN(Ws,bs,x)

	testing.test_AS(Af,f,X)


if __name__=='__main__':
	
	print(gen_AS_NN(6))
	print(gen_lossgrad_AS_NN(6,util.sqloss))

	print(gen_AS_NN(10))
	print(gen_lossgrad_AS_NN(10,util.sqloss))
