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
import pdb



allperms={n:ps.allperms(n) for n in range(9)}

activation=util.ReLU



# NN ----------------------------------------------------------------------------------------------------

@jax.jit
def apply_top_layers(params,Y):
	Ws,bs=params
	for W,b in zip(Ws,bs):
		Yb=jax.vmap(jnp.add,in_axes=(-2,0),out_axes=-2)(Y,b)
		acYb=activation(Yb)
		Y=util.apply_on_n(W,acYb)	
	return jnp.squeeze(Y)


@jax.jit
def NN(Ws,bs,X):
	L1=util.dot_nd(W0,X)
	return apply_top_layers([Ws[1:],bs],X)



###====================================================================================================
#AS ----------------------------------------------------------------------------------------------------

def gen_Af(f_top):
	@jax.jit
	def Af(W0,f_top_params,X):
		n=W0.shape[-2]						# W0:	m,n,d
		Ps,signs=allperms[n]					# Ps:	n!,n,n
		PW0=util.apply_on_n(Ps,W0)				# PW0:	n!,m,n,d
		L1=util.dot_nd(PW0,X)					# L1:	n!,m,s	
		fX=f_top(f_top_params,L1)				# fX:	n!,s
		return jnp.dot(signs,fX)/jnp.sqrt(math.factorial(n))	# s

	return Af


_AS_NN_=gen_Af(apply_top_layers)

@jax.jit
def AS_NN(Ws,bs,X):
	return _AS_NN_(Ws[0],[Ws[1:],bs],X)






####====================================================================================================
#heavy AS ----------------------------------------------------------------------------------------------


# 
# 
# def permpairs(n):
# 	n_block=min(n,6)
# 	preperms =ps.allperms(n,fix_first=n-n_block)
# 	postperms=ps.allperms(n,keep_order_of_last=n_block)
# 	return postperms,preperms
# 
# 
# 
# def gen_sumpermblock(f_top):
# 	@jax.jit	
# 	def sumpermblock(postperms,preperms,W0,f_top_params,X):
# 		postP,postsign=postperms
# 		prePs,presigns=preperms	
# 		Ps=util.apply_on_n(postP,prePs)
# 		signs=postsign*presigns
# 		PW0=util.apply_on_n(Ps,W0)			
# 		L1=util.dot_nd(PW0,X)				
# 		fX=f_top(f_top_params,L1)					
# 		return jnp.dot(signs,fX)			
# 	return sumpermblock
# 
# sumpermblock=gen_sumpermblock(apply_top_layers)
# 
# 
# def _AS_NN_(W0,f_params,X):
# 	n=W0.shape[-2]					
# 	postperms,preperms=permpairs(n)
# 	out=0
# 	for postperms in zip(postperms[0],postperms[1]):
# 		out+=sumpermblock(postperms,preperms,W0,f_params,X)
# 	return out/jnp.sqrt(math.factorial(n))
# 
# 
# def AS_NN(Ws,bs,X):
# 	return _AS_NN_(Ws[0],[Ws[1:],bs],X)
# 
# 







##---------------------------------------------------------------------------------------------------- 
## test
##---------------------------------------------------------------------------------------------------- 


def test_AS(Ws,bs,X):

	Af=lambda x:AS_NN(Ws,bs,x)
	f=lambda x:NN(Ws,bs,x)

	testing.test_AS(Af,f,X)


	
