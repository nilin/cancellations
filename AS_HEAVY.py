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
from config import session
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






def gen_Af_heavy(n,f):
	(Qs,signs_l),(Ps,signs_r)=permpairs(n)
	partial_Af=gen_partial_Af(f)

	def Af_heavy(params,X):
		PX=util.apply_on_n(Ps,X)
		out=0
		for i,(Q,sign_l) in enumerate(zip(Qs,signs_l)):
			QPX=util.apply_on_n(Q,PX)				# PX:	n!,s,n,d
			out=out+partial_Af(params,QPX,sign_l*signs_r)

			cfg.set_temp('permutation',(i+1)*PX.shape[0])
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

			cfg.set_temp('permutation',(i+1)*PX.shape[0])
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
"""
# class HeavyTrainer(learning.TrainerWithValidation):
# 
# 	# For the case when large minibatch updates are desired for to reduce noise,
# 	# but minibatch sizes are a priori restricted by memory bound. 
# 	# 
# 	#
# 	# Each sample takes significant memory,
# 	# so a minibatch can be done a few (microbatch) samples at a time
# 	# [(X_micro1,Y_micro1),(X_micro2,Y_micro2),...]
# 	# If minibatch fits in memory input [(X_minibatch,Y_minibatch)]
#
#
# 	def minibatch_step(self,X_mini,Y_mini,**kwargs):
# 
# 		microbatches=util.chop((X_mini,Y_mini),memorybatchlimit(self.n))
# 		microbatchlosses=[]
# 		microbatchparamgrads=None
# 
# 		for i,(x,y) in enumerate(microbatches):
# 
# 			grad,loss=self.lossgrad(self.weights,x,y)
# 			microbatchlosses.append(loss/self.nullloss)
# 			microbatchparamgrads=util.addgrads(microbatchparamgrads,grad)
# 
# 		updates,self.state=self.opt.update(microbatchparamgrads,self.state,self.weights)
# 		self.weights=optax.apply_updates(self.weights,updates)
# 
# 		minibatchloss=jnp.average(jnp.array(microbatchlosses))
# 		self.tracker.set('minibatch loss',minibatchloss)
"""


def memorybatchlimit(n):
	s=1
	memlim=50000
	while(s*math.factorial(n)<memlim):
		s=s*2

	if n>heavy_threshold:
		assert s==1, 'AS_HEAVY assumes single samples'

	return s



def makeblockwise(f):
	def blockwise_f(X):
		_,n,_=X.shape	
		Xs=util.chop(X,chunksize=memorybatchlimit(n))
		out=[]
		for i,(B,) in enumerate(Xs):
			out.append(f(B))
			session.trackcurrent('block',(i+1,len(Xs)))
		session.trackcurrent('block',(0,1))
		return jnp.concatenate(out,axis=0)
	return blockwise_f



