import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
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
import plottools as pt
import collections
import copy
from bookkeep import HistTracker,bgtracker





#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

collectivelossfn=util.sqloss

	
class BasicTrainer:
	def __init__(self,widths,X,Y,initfromfile=None,**kwargs):


		self.nullloss=collectivelossfn(Y,0)

		# get data and init params
		self.X,self.Y=X,Y
		self.n,self.d=X.shape[-2:]

		self.weights=genW(rnd.PRNGKey(0),self.n,self.d,widths) if initfromfile==None else self.importparams(initfromfile)

		#----------------------------------------------------------------------
		self.set_Af()
		#----------------------------------------------------------------------

		# choose training functions according to modes
		self.opt=optax.adamw(.01)
		self.state=self.opt.init(self.weights)

		self.tracker=HistTracker()
		self.tracker.track('weights',self.weights)
		self.tracker.add_autosavepath('data/hists/ID {}'.format(self.tracker.ID))
		self.tracker.add_autosavepath('data/hist')

		self.set_default_batchsizes(**kwargs)

	
	def set_Af(self):
		self.Af=AS_tools.gen_AS_NN(self.n)
		self.lossgrad=AS_tools.gen_lossgrad_AS_NN(self.n,collectivelossfn)


	def minibatch_step(self,X_mini,Y_mini,**kwargs):
	
		grad,loss=self.lossgrad(self.weights,X_mini,Y_mini)
		updates,self.state=self.opt.update(grad,self.state,self.weights)
		self.weights=optax.apply_updates(self.weights,updates)

		self.tracker.set('minibatch loss',loss)
		return loss
		

	def epoch(self,minibatchsize=None,**kwargs):

		if minibatchsize==None: minibatchsize=self.minibatchsize

		X,Y=util.randperm(self.X,self.Y)
		samples=X.shape[0]
		minibatchlosses=[]
		minibatches=util.chop((X,Y),minibatchsize)

		for i,(X_mini,Y_mini) in enumerate(minibatches):

			loss=self.minibatch_step(X_mini,Y_mini,**kwargs)	
			minibatchlosses.append(loss/self.nullloss)

			bgtracker.set('samples done',(i+1)*minibatchsize)
			bgtracker.set('minibatch losses',minibatchlosses)
		
		self.tracker.set('epoch loss',jnp.average(minibatchlosses))

	def importparams(self,path):
		self.tracker.set('event','Imported params from '+path)
		return bk.getlastval(path,'weights')

	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
		self.minibatchsize=min(self.X.shape[0],memorybatchlimit(self.n)) if minibatchsize==None else minibatchsize
		self.tracker.set('event','minibatch size set to '+str(self.minibatchsize))

	def setvals(self,**kwargs):
		self.set_default_batchsizes(**kwargs)





class TrainerWithValidation(BasicTrainer):

	def __init__(self,widths,X__,Y__,validationbatchsize=1000,fractionforvalidation=None,**kwargs):
		if fractionforvalidation!=None:
			validationbatchsize=int(X__.shape[0]*fractionforvalidation)
		trainingsamples=X__.shape[0]-validationbatchsize
		assert trainingsamples>=validationbatchsize

		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]
		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		super().__init__(widths,X_train,Y_train,**kwargs)
		self.tracker.register(self,['X','Y','X_val','Y_val','n','nullloss'])

	def validationloss(self):
		return collectivelossfn(self.Af(self.weights,self.X_val),self.Y_val)

	def checkpoint(self):
		self.tracker.set('validation loss',self.validationloss())
		self.tracker.set('weights',copy.deepcopy(self.weights))
		self.tracker.checkpoint()








class HeavyTrainer(TrainerWithValidation):

	"""
	# Each sample takes significant memory,
	# so a minibatch can be done a few (microbatch) samples at a time
	# [(X_micro1,Y_micro1),(X_micro2,Y_micro2),...]
	# If minibatch fits in memory input [(X_minibatch,Y_minibatch)]
	# """
	def minibatch_step(self,X_mini,Y_mini,**kwargs):

		microbatches=util.chop((X_mini,Y_mini),memorybatchlimit(self.n))
		microbatchlosses=[]
		microbatchparamgrads=None

		for i,(x,y) in enumerate(microbatches):

			grad,loss=self.lossgrad(self.weights,x,y)
			microbatchlosses.append(loss/self.nullloss)
			microbatchparamgrads=util.addgrads(microbatchparamgrads,grad)

		updates,self.state=self.opt.update(microbatchparamgrads,self.state,self.weights)
		self.weights=optax.apply_updates(self.weights,updates)

		self.tracker.set('training loss',loss)
		return jnp.average(jnp.array(microbatchlosses))



#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------



def memorybatchlimit(n):
	s=1
	while(s*math.factorial(n)<200000):
		s=s*2

	if n>AS_tools.heavy_threshold:
		assert s==1, 'AS_HEAVY assumes single samples'

	return s



def genW(k0,n,d,widths):

	if type(widths)!=list:
		print('Casting width to singleton list')
		widths=[widths]

	k1,*Wkeys=rnd.split(k0,100)
	k2,*bkeys=rnd.split(k0,100)

	Ws=[rnd.normal(key,(m2,m1))/math.sqrt(m1) for m1,m2,key in zip([n*d]+widths,widths+[1],Wkeys)]
	bs=[rnd.normal(key,(m,)) for m,key in zip(widths,bkeys)]

	return [Ws,bs]



def AS_from_hist(path):
	params=bk.getlastval(path,'weights')
	Af=AS_tools.gen_AS_NN(bk.getlastval(path,'n'))

	def Af_of_X(X):
		return Af(params,X)
	return Af_of_X

