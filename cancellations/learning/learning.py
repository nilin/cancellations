#
# nilin
#
# 2022/7
#






import jax
import jax.numpy as jnp
import jax.random as rnd
from ..utilities import numutil as mathutil, tracking, textutil, numutil
import optax
import math
#import universality
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import time
import pdb
from ..functions import AS_tools

import collections
import copy
from collections import deque
from ..functions import multivariate as mv


#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

	
class Trainer():
	def __init__(self,lossgrad,learner,sampler,learning_rate=.01,**kwargs):

		self.lossgrad=lossgrad
		self.learner=learner
		self.sampler=sampler

		#self.samples,self.n,self.d=X.shape
		self.opt=optax.adamw(learning_rate,**{k:val for k,val in kwargs.items() if k in ['weight_decay','mask']})
		self.state=self.opt.init(self.learner.weights)
		

	def minibatch_step(self,X_mini,*Y_mini):
		loss,grad=self.lossgrad(self.learner.weights,X_mini,*Y_mini)
		updates,self.state=self.opt.update(grad,self.state,self.learner.weights)
		self.learner.weights=optax.apply_updates(self.learner.weights,updates)
		return loss


	def step(self):
		(X_mini,*Y_mini)=self.sampler.step()
		return self.minibatch_step(X_mini,*Y_mini)	







#	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
#		self.minibatchsize=min(self.X.shape[0],cfg.memorybatchlimit(self.n),1000) if minibatchsize==None else minibatchsize
#		self.memory.log('minibatch size set to '+str(self.minibatchsize))
#
#	def checkpoint(self):
#		self.memory.remember('weights',copy.deepcopy(self.learner.weights))
#		self.memory.log('learner checkpoint')
#
#
#	def compilegrad(self):
#		self.memory.log('compiling learning gradient')	
#		X_dummy=jnp.zeros_like(self.X[:self.minibatchsize])
#		Y_dummy=jnp.zeros_like(self.Y[:self.minibatchsize])
#		self.lossgrad(self.learner.weights,X_dummy,Y_dummy)
		



#
#
#
#class DynamicTrainer(Trainer):
#	def __init__(self,learner,X,**kwargs):
#		super().__init__(learner,X,None,**kwargs)
#
#	def next_X_minibatch(self):
#		if len(self.minibatches)==0:
#			self.prepnextepoch()
#		return self.minibatches.popleft()
#
#	def step(self,f_target):
#		(X_mini,)=self.next_X_minibatch()
#		return self.minibatch_step(X_mini,f_target(X_mini))	
#
#	def prepnextepoch(self):
#		[self.X]=mathutil.randperm(self.X)
#		self.minibatches=deque(mathutil.chop(self.X,blocksize=self.minibatchsize))
#
#		self.memory.log('start new epoch')
#		self.memory.remember('minibatches in epoch',len(self.minibatches))
#
#
#class NoTargetTrainer(DynamicTrainer):
#
#	def step(self):
#		(X_mini,)=self.next_X_minibatch()
#		return self.minibatch_step(X_mini)	
#
#
#
#class Dummylearner:
#	def __init__(self,directloss,weights):
#		self.weights=weights
#		self.directloss=directloss
#
#	def get_lossgrad(self,*args,**kw):
#		return jax.value_and_grad(self.directloss)
#
#class DirectlossTrainer(NoTargetTrainer):
#	def __init__(self,directloss,weights,X,**kw):
#		super().__init__(Dummylearner(directloss,weights),X,**kw)
#
#



