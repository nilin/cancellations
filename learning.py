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





#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

individuallossfn=util.sqlossindividual
collectivelossfn=util.sqloss

	
class BasicTrainer:
	def __init__(self,widths,X,Y,initfromfile=None,**kwargs):


		# init hist
		self.epochlosses=[]
		self.paramshist=[]
		self.trainerrorhist=[]
		self.timestamps=[]
		self.nullloss=collectivelossfn(Y,0)
		self.events=[]

		# get data and init params
		self.X,self.Y=X,Y
		self.n,self.d=X.shape[-2:]
		bk.track('samples',X.shape[0])

		if initfromfile==None:
			k0=rnd.PRNGKey(0)
			self.weights=genW(k0,self.n,self.d,widths)
		else:
			self.importparams(initfromfile)

		self.ID=bk.nowstr()
		self.autosavepaths=['data/hists/started '+self.ID,'data/hist']


		#----------------------------------------------------------------------
		self.set_Af()
		#----------------------------------------------------------------------


		# choose training functions according to modes
		self.opt=optax.adamw(.01)
		self.state=self.opt.init(self.weights)

		self.setvals(**kwargs)
		self.checkpoint()
	


	
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

			#loss=self.minibatch_step(X_mini,Y_mini,**kwargs)	
			loss=self.minibatch_step_heavy(X_mini,Y_mini,**kwargs)	
			minibatchlosses.append(loss/self.nullloss)

			bk.track('minibatch losses',minibatchlosses)
			bk.track('samplesdone',(i+1)*minibatchsize)
			
		self.epochlosses.append(jnp.average(minibatchlosses))
		print()

	def epochsdone(self):
		return len(self.epochlosses)
	
	def importparams(self,path):
		self.weights=bk.get(path)['paramshist'][-1]
		self.add_event('Imported params from '+path)

	def set_default_batchsizes(self,minibatchsize=None,**kwargs):
		self.minibatchsize=min(self.X.shape[0],memorybatchlimit(self.n)) if minibatchsize==None else minibatchsize
		print('minibatch size set to '+str(self.minibatchsize))

	def setvals(self,**kwargs):
		self.set_default_batchsizes(**kwargs)
"""
#	def savehist(self,filename):
#		data={'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'valerrorhist':self.valerrorhist,'timestamps':self.timestamps,'events':self.events,'n':self.n}
#		bk.save(data,filename)
#
#	def checkpoint(self):
#		self.paramshist.append(self.weights)
#		self.trainerrorhist.append(self.epochlosses[-1] if self.epochsdone()>0 else math.nan)
#		self.timestamps.append(self.time_elapsed())
#		self.autosave()
#
#
#	def time_elapsed(self):
#		return time.perf_counter()
#	def add_event(self,msg):
#		self.events.append((self.time_elapsed(),msg))
#		bk.printemph(msg)
#
#	def autosave(self):
#		for path in self.autosavepaths:
#			self.savehist(path)
#
"""



class Tracker:

	def __init__(self,trackedvars=None,autosavepaths=[],**kwargs):
		self.trackedvars={} if trackedvars==None else trackedvars
		self.t0=time.perf_counter
		self.autosavepaths=autosavepaths
		self.activevals=dict()

	def time_elapsed(self):
		return time.perf_counter()-self.t0

	def getpassivevals(self):
		return {name:vars(self)[name] for name in self.trackedpassive}

	def set(self,name,val):
		self.activevals[name]=val

	def getallvals(self):
		return self.getpassivevals() | self.activevals

	def save(self,path):
		bk.save(data,path)
		
	def autosave(self):
		for path in self.autosavepaths:
			self.save(path)



class HistTracker(Tracker):
	
	def __init__(self,**kwargs):
		super().__init__()
		self.set_schedule(**kwargs)
		self.hist=[]
		self.events=[]
		self.individualhistories=dict()

	def set_schedule(self,**kwargs)
		self.schedule=collections.deque(list(range(60),1)+list(range(60,600,5)+list(range(600,3600,15))+list(range(3600,86400,60))))

	def poke(self):
		if self.time_elapsed>self.schedule[0]:
			self.schedule.popleft()
			self.checkpoint()
			self.save()

	def checkpoint(self):
		self.hist.append((self.time_elapsed(),self.getallvals()))

	def add_event(self,msg):
		self.events.append((self.time_elapsed(),msg))
		bk.printemph(msg)





class TrainerWithValidation(BasicTrainer,Tracker):

	def __init__(self,widths,X__,Y__,fractionforvalidation=.1,**kwargs):
		trainingsamples=round(X__.shape[0]*(1-fractionforvalidation))
		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]
		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		self.valerrorhist=[]
		super().__init__(widths,X_train,Y_train,**kwargs)

	def validationerror(self):
		vallosses=self.individuallosses(self.X_val,self.Y_val)
		bk.track('validation loss',jnp.average(vallosses)/self.nullloss)
		return vallosses

	def checkpoint(self):
		self.set('validationerror',self.validationerror())
		super().checkpoint()

	def savehist(self,filename):
		data={'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'valerrorhist':self.valerrorhist,'timestamps':self.timestamps,'events':self.events,'n':self.n}
		bk.save(data,filename)

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
	params=bk.get(path)['paramshist'][-1]
	Af=AS_tools.gen_AS_NN(bk.get(path)['n'])

	def Af_of_X(X):
		return Af(params,X)
	return Af_of_X

def losses_from_hist(path):
	out={k:v for k,v in bk.get(path).items() if k in {'timestamps','trainerrorhist'}}
	out['valerrorhist']=[jnp.average(_) for _ in bk.get(path)['valerrorhist']]
	return {k:jnp.array(v) for k,v in out.items()}
	



#----------------------------------------------------------------------------------------------------
# losses and gradients
#----------------------------------------------------------------------------------------------------






#
#
#
#@jax.jit
#def individuallossesNS(Wb,X,Y):
#	Z=AS_tools.NN(Wb,X)
#	return individualloss(Y,Z)
#
#@jax.jit
#def collectivelossNS(Wb,X,Y):
#	return jnp.average(individuallossesNS(Wb,X,Y))
#
#@jax.jit
#def lossgradNS(Wb,X,Y):
#	loss,grad=jax.value_and_grad(collectivelossNS)(Wb,X,Y)
#	return grad,loss
#
#
