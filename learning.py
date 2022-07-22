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


	
class Trainer:
	def __init__(self,widths,X,Y,batchmode='minibatch',initfromfile=None):

		# init hist
		self.epochlosses=[]
		self.paramshist=[]
		self.trainerrorhist=[]
		self.timestamps=[]
		self.nullloss=collectiveloss(Y,0)
		self.events=[]

		# get data and init params
		self.n,self.d=X.shape[-2:]
		self.X,self.Y=X,Y

		if initfromfile==None:
			k0=rnd.PRNGKey(0)
			self.weights=genW(k0,self.n,self.d,widths)
		else:
			self.importparams(initfromfile)



		# choose training functions according to modes
		self.set_symmode('AS')
		self.set_batchmode(batchmode)

		self.ID=bk.nowstr()
		self.checkpoint()
		




	def minibatch_epoch(self,minibatchsize):
		
		X,Y=util.randperm(self.X,self.Y)
		samples=X.shape[0]
		minibatchlosses=[]

		for a in range(0,samples,minibatchsize):
			c=min(a+minibatchsize,samples)

			x=X[a:c]
			y=Y[a:c]

			grad,loss=self.lossgrad(self.weights,x,y)

			updates,self.state=self.opt.update(grad,self.state,self.weights)
			self.weights=optax.apply_updates(self.weights,updates)

			rloss=loss/self.nullloss
			minibatchlosses.append(rloss)
			bk.printbar(rloss,msg='training loss  ',style=bk.box)
		print()
		self.epochlosses.append(jnp.average(minibatchlosses))


	def batch_epoch(self,minibatchsize):
	
		X,Y=self.X,self.Y	
		samples=X.shape[0]
		minibatchlosses=[]
		minibatchparamgrads=None

		print('\n\n')
		for a in range(0,samples,minibatchsize):
			c=min(a+minibatchsize,samples)

			x=X[a:c]
			y=Y[a:c]

			grad,loss=self.lossgrad(self.weights,x,y)


			rloss=loss/self.nullloss
			minibatchlosses.append(rloss)
			minibatchparamgrads=addparamgrads(minibatchparamgrads,grad)

			completeness=1.0*c/samples; bk.printbar(completeness,msg='Epoch '+str(round(completeness*100))+'% complete',printval=False,style=10*bk.bar+'minibatches done',emptystyle=' ')

		updates,self.state=self.opt.update(minibatchparamgrads,self.state,self.weights)
		self.weights=optax.apply_updates(self.weights,updates)

		print('Parameter update'+200*' ')
		bk.printbar(rloss,msg='training loss  ',style='\u2592',hold=False)
		self.epochlosses.append(jnp.average(minibatchlosses))



	def set_symmode(self,mode):
		self.lossgrad=lossgradAS if mode=='AS' else lossgradNS
		self.individuallosses=individuallossesAS if mode=='AS' else individuallossesNS


	def set_batchmode(self,batchmode):
		self.epoch={'batch':self.batch_epoch,'minibatch':self.minibatch_epoch}[batchmode]
		self.add_event('Batch mode : '+batchmode)
		self.set_optstate(batchmode)

	def set_optstate(self,batchmode):
		self.opt=optax.adamw({'batch':.001,'minibatch':.01}[batchmode])
		self.state=self.opt.init(self.weights)

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
		


def addparamgrads(G1,G2):

	if G1==None:
		return G2
	elif type(G2)==list:
		return [addparamgrads(g1,g2) for g1,g2 in zip(G1,G2)]
	else:
		return G1+G2
		




class TrainerWithValidation(Trainer):

	def __init__(self,widths,X__,Y__,fractionforvalidation=.1,**kwargs):
		trainingsamples=round(X__.shape[0]*(1-fractionforvalidation))
		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]

		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		self.valerrorhist=[]
		super().__init__(widths,X_train,Y_train,**kwargs)

	def validationerror(self,loud=False):
		vallosses=self.individuallosses(self.weights,self.X_val,self.Y_val)/self.nullloss
		if loud:
			bk.printbar(jnp.average(vallosses),msg='validation loss',hold=False)
		return vallosses

	def checkpoint(self):
		self.valerrorhist.append(self.validationerror(loud=True))
		super().checkpoint()

	def savehist(self,filename):
		data={'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'valerrorhist':self.valerrorhist,'timestamps':self.timestamps,'events':self.events}
		bk.save(data,filename)

	def autosave(self):
		self.savehist('data/hists/started '+self.ID)


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
def initandtrain(data_in_path,data_out_path,widths,batchsize,action_each_epoch=donothing,action_on_pause=donothing,**kwargs): 
	X,Y=bk.get(data_in_path)
	T=TrainerWithValidation(widths,X,Y,**kwargs)
	try:
		while True:
			try:
				print('\nEpoch '+str(T.epochsdone()))
				T.epoch(batchsize)
				action_each_epoch(T)
			except KeyboardInterrupt:
				action_on_pause(T)
				continue
			T.savehist(data_out_path)
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
	@jax.jit
	def Af(X):
		return AS_tools.AS_NN(params,X)
	return Af

def losses_from_hist(path):
	out={k:v for k,v in bk.get(path).items() if k in {'timestamps','trainerrorhist'}}
	out['valerrorhist']=[jnp.average(_) for _ in bk.get(path)['valerrorhist']]
	return {k:jnp.array(v) for k,v in out.items()}
	



#----------------------------------------------------------------------------------------------------
# losses and gradients
#----------------------------------------------------------------------------------------------------




individualloss=util.sqlossindividual
collectiveloss=util.sqloss


@jax.jit
def individuallossesAS(params,X,Y):
	Z=AS_tools.AS_NN(params,X)
	return individualloss(Y,Z)

@jax.jit
def collectivelossAS(Wb,X,Y):
	return jnp.average(individuallossesAS(Wb,X,Y))

@jax.jit
def lossgradAS(params,X,Y):
	loss,grad=jax.value_and_grad(collectivelossAS)(params,X,Y)
	return grad,loss



@jax.jit
def individuallossesNS(Wb,X,Y):
	Z=AS_tools.NN(Wb,X)
	return individualloss(Y,Z)

@jax.jit
def collectivelossNS(Wb,X,Y):
	return jnp.average(individuallossesNS(Wb,X,Y))

@jax.jit
def lossgradNS(Wb,X,Y):
	loss,grad=jax.value_and_grad(collectivelossNS)(Wb,X,Y)
	return grad,loss


