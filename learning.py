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
import multivariate as mv




#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------


	
class Trainer():
	def __init__(self,learner,X,Y,lossgrad=None,learner_lossfn_choice=0,learning_rate=.01,memory=None,**kwargs):

		self.memory=cfg.Memory() if memory==None else memory

		self.learner=learner
		self.lossgrad=learner.lossgrads[learner_lossfn_choice] if lossgrad==None else lossgrad

		self.X,self.Y=X,Y
		self.samples,self.n,self.d=X.shape

		self.opt=optax.adamw(learning_rate,**{k:val for k,val in kwargs.items() if k in ['weight_decay','mask']})
		self.state=self.opt.init(self.learner.weights)

		self.set_default_batchsizes(**kwargs)
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


	def prepnextepoch(self):
		self.X,self.Y=util.randperm(self.X,self.Y)
		self.minibatches=deque(util.chop(self.X,self.Y,chunksize=self.minibatchsize))

		self.memory.log('start new epoch')
		self.memory.remember('minibatches in epoch',len(self.minibatches))


	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
		self.minibatchsize=min(self.X.shape[0],AS_HEAVY.memorybatchlimit(self.n),1000) if minibatchsize==None else minibatchsize
		self.memory.log('minibatch size set to '+str(self.minibatchsize))

	def get_learned(self):
		return self.learner.as_static()

	def checkpoint(self):
		self.memory.remember('weights',copy.deepcopy(self.learner.weights))
		self.memory.log('learner checkpoint')

#	def save(self):
#		self.checkpoint()
#		cfg.autosave()





class DynamicTrainer(Trainer):
	def __init__(self,learner,X,**kwargs):
		super().__init__(learner,X,None,**kwargs)

	def next_X_minibatch(self):
		if len(self.minibatches)==0:
			self.prepnextepoch()
		cfg.trackcurrent('minibatches left',len(self.minibatches))
		return self.minibatches.popleft()

	def step(self,f_target):
		(X_mini,)=self.next_X_minibatch()
		return self.minibatch_step(X_mini,f_target(X_mini))	

	def prepnextepoch(self):
		[self.X]=util.randperm(self.X)
		self.minibatches=util.chop(self.X,chunksize=self.minibatchsize)
		self.minibatches=deque(self.minibatches)

		cfg.log('start new epoch')
		cfg.remember('minibatchsize',self.minibatchsize)
		cfg.trackcurrent('minibatches',len(self.minibatches))


class NoTargetTrainer(DynamicTrainer):

	def step(self):
		(X_mini,)=self.next_X_minibatch()
		return self.minibatch_step(X_mini)	


#----------------------------------------------------------------------------------------------------

class Learner:
	def __init__(self,f,lossgrads=None,weights=None,deepcopy=True):
		self.f=f
		self.lossgrads=[] if lossgrads==None else lossgrads
		if deepcopy:
			self.reset(weights)
		else:
			self.weights=weights


	def reset(self,weights):
		self.weights=copy.deepcopy(weights)
		return self

	def as_static(self):
		#return util.fixparams(self.f,self.weights)
		return AS_HEAVY.makeblockwise(util.fixparams(self.f,self.weights))

	def cloneweights(self):
		return copy.deepcopy(self.weights)



class AS_Learner(Learner):
	def __init__(self,*args,NS=None,weights=None,**kwargs):
		super().__init__(*args,weights=weights,**kwargs)
		self.NS=NS

	def get_NS(self):
		return NS_Learner(self.NS,weights=self.weights)


class NS_Learner(Learner):
	def get_AS(self):
		return AS_Learner(AS_tools.gen_Af(self.f),lossgrads=[AS_tools.gen_lossgrad_Af(self.f)],NS=self.f,weights=self.weights)


def static_NS(learner):
	if type(learner) in {NS_Learner,Learner}:
		return learner.as_static()
	if type(learner)==AS_Learner:
		return learner.get_NS().as_static()
	


#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------



