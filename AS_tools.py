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



activation=util.ReLU



#=======================================================================================================
# NN 
#=======================================================================================================



@jax.jit
def NN(params,X):
	Ws,bs=params
	for W,b in zip(Ws[:-1],bs):
		X=jnp.inner(X,W)+b[None,:]
		X=activation(X)
	return jnp.squeeze(jnp.inner(X,Ws[-1]))


@jax.jit
def NN_NS(params,X):
	X=util.collapselast(X,2)
	return NN(params,X)





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


from AS_HEAVY import gen_Af_heavy,gen_lossgrad_Af_heavy,heavy_threshold



def gen_AS_NN(n):
	return gen_Af(n,NN_NS) if n<=heavy_threshold else gen_Af_heavy(n,NN_NS)



def gen_lossgrad_AS_NN(n,lossfn):
	return gen_lossgrad_Af(n,NN_NS,lossfn) if n<=heavy_threshold else gen_lossgrad_Af_heavy(n,NN_NS,lossfn)
		





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
