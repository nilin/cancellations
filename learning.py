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
import AS_functions
import AS_HEAVY
import plottools as pt
import collections
import copy
from collections import deque
import multivariate




#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

collectivelossfn=util.sqloss

	
class Trainer(cfg.State):
	def __init__(self,learner,X,Y,**kwargs):

		self.learner=learner

		self.nullloss=collectivelossfn(Y,0)
		self.X,self.Y=X,Y
		self.samples,self.n,self.d=X.shape

		self.opt=optax.adamw(.01)
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
	def __init__(self,*args,weights=None):
		self.f,self.lossgrad,*_=args
		self.reset(weights)

	def reset(self,weights):
		self.weights=weights
		return self

	def as_static(self):
		return util.fixparams(self.f,self.weights)



class AS_Learner(Learner):
	def __init__(self,*args,weights=None):
		super().__init__(*args,weights)
		self.NS=args[-1]

	def static_NS(self):
		return util.fixparams(self.NS,self.weights)




#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------



