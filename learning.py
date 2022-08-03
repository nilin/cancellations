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


	
class Trainer(cfg.State):
	def __init__(self,learner,X,Y,learning_rate=.01,**kwargs):

		self.learner=learner

		self.X,self.Y=X,Y
		self.samples,self.n,self.d=X.shape

		self.opt=optax.adamw(learning_rate,**{k:val for k,val in kwargs.items() if k in ['weight_decay','mask']})
		self.state=self.opt.init(self.learner.weights)

		self.set_default_batchsizes(**kwargs)
		self.minibatches=deque([])

		super().__init__()
		cfg.setstatic('weightshistpointer',self.linkentry('weights'))


	def minibatch_step(self,X_mini,Y_mini):
	
		loss,grad=self.learner.lossgrad(self.learner.weights,X_mini,Y_mini)
		updates,self.state=self.opt.update(grad,self.state,self.learner.weights)
		self.learner.weights=optax.apply_updates(self.learner.weights,updates)

		cfg.remember('minibatch loss',loss)
		self.remember('minibatch loss',loss)
		self.remember('total steps done',len(self.gethist('minibatch loss')[0]))


	def step(self):
		if len(self.minibatches)==0:
			self.prepnextepoch()
		(X_mini,Y_mini)=self.minibatches.popleft()
		self.minibatch_step(X_mini,Y_mini)	
		cfg.trackcurrent('minibatches left',len(self.minibatches))


	def prepnextepoch(self):
		self.X,self.Y=util.randperm(self.X,self.Y)
		#self.minibatches=deque(util.chop((self.X,self.Y),self.minibatchsize))
		self.minibatches=deque(util.chop(self.X,self.Y,chunksize=self.minibatchsize))

		cfg.log('start new epoch')
		cfg.remember('minibatchsize',self.minibatchsize)
		cfg.trackcurrent('minibatches',len(self.minibatches))


	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
		self.minibatchsize=min(self.X.shape[0],AS_HEAVY.memorybatchlimit(self.n),1000) if minibatchsize==None else minibatchsize
		cfg.log('minibatch size set to '+str(self.minibatchsize))


	def get_learned(self):
		return self.learner.as_static()

	def checkpoint(self):
		cfg.remember('weights',copy.deepcopy(self.learner.weights))
		self.remember('weights',copy.deepcopy(self.learner.weights))
		cfg.log('learner checkpoint')

	def save(self):
		self.checkpoint()
		cfg.autosave()





#----------------------------------------------------------------------------------------------------

class Learner:
	def __init__(self,f,lossgrad=None,weights=None,deepcopy=True):
		self.f=f
		self.lossgrad=mv.gen_lossgrad(f) if lossgrad==None else lossgrad
		if deepcopy:
			self.reset(weights)
		else:
			self.weights=weights


	def reset(self,weights):
		self.weights=copy.deepcopy(weights)
		return self

	def as_static(self):
		return util.fixparams(self.f,self.weights)

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
		return AS_Learner(AS_tools.gen_Af(self.f),lossgrad=AS_tools.gen_lossgrad_Af(self.f),NS=self.f,weights=self.weights)


def static_NS(learner):
	if type(learner) in {NS_Learner,Learner}:
		return learner.as_static()
	if type(learner)==AS_Learner:
		return learner.get_NS().as_static()
	


#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------



