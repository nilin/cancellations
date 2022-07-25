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
from bookkeep import HistTracker




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
		self.set('samples',X.shape[0])

		self.weights=genW(rnd.PRNGKey(0),self.n,self.d,widths) if initfromfile==None else self.importparams(initfromfile)

		#----------------------------------------------------------------------
		self.set_Af()
		#----------------------------------------------------------------------

		# choose training functions according to modes
		self.opt=optax.adamw(.01)
		self.state=self.opt.init(self.weights)

		self.set_default_batchsizes(**kwargs)


	
	def set_Af(self):
		self.Af=AS_tools.gen_AS_NN(self.n)
		self.lossgrad=AS_tools.gen_lossgrad_AS_NN(self.n,collectivelossfn)




	def minibatch_step(self,X_mini,Y_mini,**kwargs):
	
		grad,loss=self.lossgrad(self.weights,X_mini,Y_mini)
		updates,self.state=self.opt.update(grad,self.state,self.weights)
		self.weights=optax.apply_updates(self.weights,updates)

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

			#bk.track('minibatch losses',minibatchlosses)
			#bk.track('samplesdone',(i+1)*minibatchsize)
			self.set('minibatch losses',minibatchlosses)
			self.set('samples done',(i+1)*minibatchsize)
		
		self.set('epoch loss',jnp.average(minibatchlosses))

	def importparams(self,path):
		self.add_event('Imported params from '+path)
		return bk.getlastval(path,'weights')

	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
		self.minibatchsize=min(self.X.shape[0],memorybatchlimit(self.n)) if minibatchsize==None else minibatchsize
		print('minibatch size set to '+str(self.minibatchsize))







class TrainerWithValidation(BasicTrainer):

	def __init__(self,widths,X__,Y__,validationbatchsize=1000,fractionforvalidation=None,**kwargs):
		if fractionforvalidation!=None:
			validationbatchsize=int(X__.shape[0]*fractionforvalidation)
		trainingsamples=X__.shape[0]-validationbatchsize
		assert trainingsamples>=validationbatchsize

		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]
		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		super().__init__(widths,X_train,Y_train,**kwargs)

	def validationloss(self):
		return collectivelossfn(self.Af(self.weights,self.X_val),self.Y_val)




class Trainer(TrainerWithValidation,HistTracker):

	def __init__(self,*args,**kwargs):
		HistTracker.__init__(self)
		super().__init__(*args,**kwargs)
		self.register('X','Y','X_val','Y_val','n','nullloss')
		self.track('weights')
		self.add_autosavepath('data/hists/ID {}'.format(self.ID))
		self.add_autosavepath('data/hist')

	def checkpoint(self):
		self.set('validation loss',self.validationloss())
		HistTracker.checkpoint(self)












"""
#	def makingprogress(self,p_val=.10):
#		return True if len(self.valerrorhist)<2 else util.distinguishable(self.valerrorhist[-2],self.valerrorhist[-1],p_val,alternative='greater')
#
#	def stale(self,p_val=.10):
#		return not self.makingprogress(p_val)
"""




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

