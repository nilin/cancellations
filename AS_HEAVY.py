# nilin

#=======================================================================================================
# heavy (n>9) 
# 
#=======================================================================================================

from jax.config import config
#config.update("jax_enable_x64", True)
import numpy as np
import math
import pickle
import time
import config as cfg
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
heavy_threshold=8

  
def permpairs(n):
 	n_block=min(n,heavy_threshold)
 	rightperms=ps.allperms(n,fix_first=n-n_block)
 	leftperms=ps.allperms(n,keep_order_of_last=n_block)
 	return leftperms,rightperms


def gen_partial_Af(f):
	@jax.jit
	def partial_Af(params,PX,signs):
		fX=f(params,PX)						# fX:	n!,s
		return jnp.squeeze(jnp.inner(signs,fX))			# s

	return partial_Af


def gen_partial_grad_Af(f):
	partial_Af=gen_partial_Af(f)
	return jax.jit(jax.value_and_grad(partial_Af))


class EmptyTracker:
	def set(self,*args):
		pass



emptytracker=EmptyTracker()



def gen_Af_heavy(n,f):
	(Qs,signs_l),(Ps,signs_r)=permpairs(n)
	partial_Af=gen_partial_Af(f)

	def Af_heavy(params,X):
		PX=util.apply_on_n(Ps,X)
		out=0
		for i,(Q,sign_l) in enumerate(zip(Qs,signs_l)):
			QPX=util.apply_on_n(Q,PX)				# PX:	n!,s,n,d
			out=out+partial_Af(params,QPX,sign_l*signs_r)

			cfg.bgtracker.set('permutation',(i+1)*PX.shape[0])
		return out

	return Af_heavy






def gen_grad_Af_heavy(n,f):
	(Qs,signs_l),(Ps,signs_r)=permpairs(n)
	partial_grad_Af=gen_partial_grad_Af(f)

	def grad_Af_heavy(params,X):
		PX=util.apply_on_n(Ps,X)
		grad,out=None,0
		for i,(Q,sign_l) in enumerate(zip(Qs,signs_l)):
			QPX=util.apply_on_n(Q,PX)				# PX:	n!,s,n,d
			blocksum,gradblocksum=partial_grad_Af(params,QPX,sign_l*signs_r)
			out=out+blocksum
			grad=util.addgrads(grad,gradblocksum)

			cfg.bgtracker.set('permutation',(i+1)*PX.shape[0])
		return grad,out

	return grad_Af_heavy




def gen_lossgrad_Af_heavy(n,f,lossfn,**kwargs):

	grad_Af_heavy=gen_grad_Af_heavy(n,f,**kwargs)

	def lossgrad_singlesample(params,x,y):
	
		assert x.shape[0]==1, 'Af_heavy is for large n and microbatchsize=1'
	
		grad,fx=grad_Af_heavy(params,x)
		loss,dloss=jax.value_and_grad(lossfn)(fx,y)

		return tuple([util.scalegrad(grad,dloss),loss])

	return lossgrad_singlesample





####################################################################################################
#
