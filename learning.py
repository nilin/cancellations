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





#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

individuallossfn=util.sqlossindividual
collectivelossfn=util.sqloss

	
class Trainer:
	def __init__(self,widths,X,Y,initfromfile=None):


		# init hist
		self.epochlosses=[]
		self.paramshist=[]
		self.trainerrorhist=[]
		self.timestamps=[]
		self.nullloss=collectivelossfn(Y,0)
		self.events=[]

		# get data and init params
		self.n,self.d=X.shape[-2:]
		self.X,self.Y=X,Y
		if initfromfile==None:
			k0=rnd.PRNGKey(0)
			self.weights=genW(k0,self.n,self.d,widths)
		else:
			self.importparams(initfromfile)
		self.set_default_batchsizes()

		self.ID=bk.nowstr()
		self.autosavepaths=['data/hists/started '+self.ID,'data/hist']


		#----------------------------------------------------------------------
		self.set_Af()
		#----------------------------------------------------------------------


		# choose training functions according to modes
		self.opt=optax.adamw(.01)
		self.state=self.opt.init(self.weights)

		self.checkpoint()
	


	
	def set_Af(self):
		self.Af=AS_tools.gen_AS_NN(self.n)
		self.lossgrad=AS_tools.gen_lossgrad_AS_NN(self.n,collectivelossfn)


	"""
	# Each sample takes significant memory,
	# so a minibatch can be done a few (microbatch) samples at a time
	# [(X_micro1,Y_micro1),(X_micro2,Y_micro2),...]
	# If minibatch fits in memory input [(X_minibatch,Y_minibatch)]
	"""
	def minibatch_step(self,minibatch_as_microbatches):
	
		microbatchlosses=[]
		microbatchparamgrads=None


		t0=time.perf_counter()

		for i,(x,y) in enumerate(minibatch_as_microbatches):

			grad,loss=self.lossgrad(self.weights,x,y)
			microbatchlosses.append(loss/self.nullloss)
			microbatchparamgrads=util.addgrads(microbatchparamgrads,grad)

			completeness=1.0*i/len(minibatch_as_microbatches)
			if time.perf_counter()-t0>.5:			
				bk.printbar(completeness,printval=False,style='minibatch '+str(round(completeness*100))+'% complete.    ',emptystyle=' ')
		
		updates,self.state=self.opt.update(microbatchparamgrads,self.state,self.weights)
		self.weights=optax.apply_updates(self.weights,updates)

		return jnp.average(jnp.array(microbatchlosses))
		

	def epoch(self,minibatchsize=None,microbatchsize=None):

		if minibatchsize==None: minibatchsize=self.minibatchsize
		if microbatchsize==None: microbatchsize=self.microbatchsize

		X,Y=util.randperm(self.X,self.Y)
		samples=X.shape[0]
		minibatchlosses=[]
		minibatches=util.chop((X,Y),minibatchsize)

		for X_mini,Y_mini in minibatches:
			microbatches=util.chop((X_mini,Y_mini),microbatchsize)
			loss=self.minibatch_step(microbatches)	
			minibatchlosses.append(loss/self.nullloss)
			bk.printbar(loss/self.nullloss,msg='training loss (minibatch) ',style=bk.box)
			
		self.epochlosses.append(jnp.average(minibatchlosses))
		print()



	def checkpoint(self):
		self.paramshist.append(self.weights)
		self.trainerrorhist.append(self.epochlosses[-1] if self.epochsdone()>0 else math.nan)
		self.timestamps.append(self.time_elapsed())
		self.autosave()


	def time_elapsed(self):
		return time.perf_counter()
		

	def epochsdone(self):
		return len(self.epochlosses)

	
	def add_event(self,msg):
		self.events.append((self.time_elapsed(),msg))
		bk.printemph(msg)

	def importparams(self,path):
		self.weights=bk.get(path)['paramshist'][-1]
		self.add_event('Imported params from '+path)


	def set_default_batchsizes(self,minibatchsize=250,microbatchsize=None):
		self.minibatchsize=min(minibatchsize,self.X.shape[0])
		self.microbatchsize=min(self.minibatchsize,microbatchsizechoice(self.n))	
		print('minibatch size set to '+str(self.minibatchsize))
		print('microbatch size set to '+str(self.microbatchsize))

		
	def autosave(self):
		for path in self.autosavepaths:
			self.savehist(path)






def microbatchsizechoice(n):
	s=1
	while(s*math.factorial(n)<10**4):
		s=s*10
	return s






class TrainerWithValidation(Trainer):

	def __init__(self,widths,X__,Y__,fractionforvalidation=.1,**kwargs):
		trainingsamples=round(X__.shape[0]*(1-fractionforvalidation))
		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]

		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		self.valerrorhist=[]
		super().__init__(widths,X_train,Y_train,**kwargs)


	"""
	# for validation error
	# not optimized for grad (take self.lossgrad instead)
	"""
	def individuallosses(self,X,Y):
		return individuallossfn(self.Af(self.weights,X),Y)

	def validationerror(self,loud=False):
		vallosses=self.individuallosses(self.X_val,self.Y_val)/self.nullloss
		if loud:
			bk.printbar(jnp.average(vallosses),msg='validation loss',hold=False)
		return vallosses

	def checkpoint(self):
		self.valerrorhist.append(self.validationerror(loud=True))
		super().checkpoint()

	def savehist(self,filename):
		data={'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'valerrorhist':self.valerrorhist,'timestamps':self.timestamps,'events':self.events,'n':self.n}
		bk.save(data,filename)


	def makingprogress(self,p_val=.10):
		return True if len(self.valerrorhist)<2 else distinguishable(self.valerrorhist[-2],self.valerrorhist[-1],p_val)

	def stale(self,p_val=.10):
		return not self.makingprogress(p_val)



	
def distinguishable(x,y,p_val=.10):
	u,p=st.mannwhitneyu(x,y,alternative='greater')
	return p<p_val




#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------


def donothing(*args):
	pass


"""
press Ctrl-C to stop training
stopwhenstale either False (no stop) or p-value (smaller means earlier stopping)
"""
def initandtrain(data_in_path,widths,action_each_epoch=donothing,action_on_pause=donothing,**kwargs): 
	X,Y=bk.get(data_in_path)
	T=TrainerWithValidation(widths,X,Y,**kwargs)
	try:
		while True:
			try:
				print('\nEpoch '+str(T.epochsdone()))
				T.epoch()
				action_each_epoch(T)
			except KeyboardInterrupt:
				action_on_pause(T)
				continue
	except KeyboardInterrupt:
		print('\nEnding.\n')



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

	@jax.jit
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
