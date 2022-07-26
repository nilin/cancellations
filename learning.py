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
from config import HistTracker
import multivariate




#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

collectivelossfn=util.sqloss

	
class BasicTrainer:
	def __init__(self,__Af__,X,Y,tracker=None,**kwargs):

		
		#----------------------------------------------------------------------
		self.Af,self.lossgrad,self.weights=__Af__		
		#----------------------------------------------------------------------

		self.nullloss=collectivelossfn(Y,0)
		self.X,self.Y=X,Y
		self.samples,self.n,self.d=X.shape

		self.tracker=HistTracker() if tracker==None else tracker
		self.tracker.add_autosavepaths('data/hist','data/hists/'+self.tracker.ID)

		self.opt=optax.adamw(.01)
		self.state=self.opt.init(self.weights)

		self.set_default_batchsizes(**kwargs)
		self.minibatches=deque([])

		return self.tracker


	def minibatch_step(self,X_mini,Y_mini):
	
		loss,grad=self.lossgrad(self.weights,X_mini,Y_mini)
		updates,self.state=self.opt.update(grad,self.state,self.weights)
		self.weights=optax.apply_updates(self.weights,updates)

		self.tracker.set('minibatch loss',loss)


	def step(self):
		if len(self.minibatches)==0:
			self.prepnextepoch()
		(X_mini,Y_mini)=self.minibatches.popleft()
		self.minibatch_step(X_mini,Y_mini)	
		self.tracker.set('minibatches left',len(self.minibatches))


	def prepnextepoch(self):
		self.X,self.Y=util.randperm(self.X,self.Y)
		self.minibatches=deque(util.chop((self.X,self.Y),self.minibatchsize))

		self.tracker.log('start new epoch')
		self.tracker.set('minibatchsize',self.minibatchsize)
		self.tracker.set('minibatches',len(self.minibatches))


	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
		self.minibatchsize=min(self.X.shape[0],memorybatchlimit(self.n)) if minibatchsize==None else minibatchsize
		self.tracker.log('minibatch size set to '+str(self.minibatchsize))

	def setvals(self,**kwargs):
		self.set_default_batchsizes(**kwargs)





class TrainerWithValidation(BasicTrainer):

	def __init__(self,__Af__,X__,Y__,validationbatchsize=1000,fractionforvalidation=None,**kwargs):
		if fractionforvalidation!=None:
			validationbatchsize=int(X__.shape[0]*fractionforvalidation)
		trainingsamples=X__.shape[0]-validationbatchsize
		assert trainingsamples>=validationbatchsize

		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]
		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		super().__init__(__Af__,X_train,Y_train,**kwargs)
		self.tracker.register(self,['X','Y','X_val','Y_val','n','nullloss'])

	def validation(self):
		validationloss=collectivelossfn(self.Af(self.weights,self.X_val),self.Y_val)
		self.tracker.set('validation loss',validationloss)
		self.tracker.log('validation set')

	def save(self):
		self.tracker.set('weights',copy.deepcopy(self.weights))
		self.tracker.autosave()
		self.tracker.log('Saved weights.')








#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------



def memorybatchlimit(n):
	s=1
	memlim=200000
	while(s*math.factorial(n)<memlim):
		s=s*2

	if n>AS_HEAVY.heavy_threshold:
		assert s==1, 'AS_HEAVY assumes single samples'

	return s





