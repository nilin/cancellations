#
# nilin
#
# 2022/7
#






import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
import optax
import math
#import universality
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import time
import pdb
import AS_tools
import AS_HEAVY
import collections
import copy
from collections import deque
from functions import multivariate as mv


#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------


	
class Trainer():
	def __init__(self,learner,X,Y,lossfn=None,learning_rate=.01,memory=None,minibatchsize=100,**kwargs):

		self.memory=cfg.Memory() if memory==None else memory

		self.learner=learner
		self.lossgrad=learner.get_lossgrad(lossfn)

		self.X,self.Y=X,Y
		self.samples,self.n,self.d=X.shape

		self.opt=optax.adamw(learning_rate,**{k:val for k,val in kwargs.items() if k in ['weight_decay','mask']})
		self.state=self.opt.init(self.learner.weights)

		self.minibatchsize=100
		#self.set_default_batchsizes(**kwargs)
		self.minibatches=deque([])

		

	def minibatch_step(self,X_mini,*Y_mini):
	
		loss,grad=self.lossgrad(self.learner.weights,X_mini,*Y_mini)
		updates,self.state=self.opt.update(grad,self.state,self.learner.weights)
		self.learner.weights=optax.apply_updates(self.learner.weights,updates)
		self.memory.remember('minibatch loss',loss)
		return loss


	def step(self):
		if len(self.minibatches)==0:
			self.prepnextepoch()
		(X_mini,Y_mini)=self.minibatches.popleft()

		self.memory.addcontext('minibatches left in epoch',len(self.minibatches))
		return self.minibatch_step(X_mini,Y_mini)	


	def prepnextepoch(self,permute=True):
		self.memory.log('preparing new epoch')
		if permute: self.X,self.Y=util.randperm(self.X,self.Y)
		self.minibatches=deque(util.chop(self.X,self.Y,blocksize=self.minibatchsize))

		self.memory.log('start new epoch')
		self.memory.remember('minibatches in epoch',len(self.minibatches))


#	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
#		self.minibatchsize=min(self.X.shape[0],cfg.memorybatchlimit(self.n),1000) if minibatchsize==None else minibatchsize
#		self.memory.log('minibatch size set to '+str(self.minibatchsize))

	def checkpoint(self):
		self.memory.remember('weights',copy.deepcopy(self.learner.weights))
		self.memory.log('learner checkpoint')


#	def compilegrad(self):
#		self.memory.log('compiling learning gradient')	
#		X_dummy=jnp.zeros_like(self.X[:self.minibatchsize])
#		Y_dummy=jnp.zeros_like(self.Y[:self.minibatchsize])
#		self.lossgrad(self.learner.weights,X_dummy,Y_dummy)
		






class DynamicTrainer(Trainer):
	def __init__(self,learner,X,**kwargs):
		super().__init__(learner,X,None,**kwargs)

	def next_X_minibatch(self):
		if len(self.minibatches)==0:
			self.prepnextepoch()
		return self.minibatches.popleft()

	def step(self,f_target):
		(X_mini,)=self.next_X_minibatch()
		return self.minibatch_step(X_mini,f_target(X_mini))	

	def prepnextepoch(self):
		[self.X]=util.randperm(self.X)
		self.minibatches=deque(util.chop(self.X,blocksize=self.minibatchsize))

		self.memory.log('start new epoch')
		self.memory.remember('minibatches in epoch',len(self.minibatches))


class NoTargetTrainer(DynamicTrainer):

	def step(self):
		(X_mini,)=self.next_X_minibatch()
		return self.minibatch_step(X_mini)	



class Dummylearner:
	def __init__(self,directloss,weights):
		self.weights=weights
		self.directloss=directloss

	def get_lossgrad(self,*args,**kw):
		return jax.value_and_grad(self.directloss)

class DirectlossTrainer(NoTargetTrainer):
	def __init__(self,directloss,weights,X,**kw):
		super().__init__(Dummylearner(directloss,weights),X,**kw)





