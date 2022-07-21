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
	def __init__(self,widths,X,Y,batchmode='minibatch'):

		self.n,self.d=X.shape[-2:]
		self.X,self.Y=X,Y
		k0=rnd.PRNGKey(0)
		self.Ws,self.bs=genW(k0,self.n,self.d,widths)

		# init hist
		self.epochlosses=[]
		self.paramshist=[]
		self.trainerrorhist=[]
		self.timestamps=[]
		self.nullloss=collectiveloss(Y,0)

		# choose training functions according to modes
		self.set_symmode('AS')
		self.set_batchmode(batchmode)
		


	def set_symmode(self,mode):
		self.lossgrad=lossgradAS if mode=='AS' else lossgradNS
		self.individuallosses=individuallossesAS if mode=='AS' else individuallossesNS
		bk.printemph('Symmetry mode : '+mode)


	def set_batchmode(self,batchmode):
		self.epoch={'batch':self.batch_epoch,'minibatch':self.minibatch_epoch}[batchmode]
		bk.printemph('Batch mode : '+batchmode)
		self.set_optstate(batchmode)

	def set_optstate(self,batchmode):
		self.opt=optax.adamw({'batch':.001,'minibatch':.01}[batchmode])
		self.state=self.opt.init([self.Ws,self.bs])

	def checkpoint(self):
		self.paramshist.append([self.Ws,self.bs])
		self.trainerrorhist.append(self.epochlosses[-1] if self.epochsdone()>0 else math.nan)
		self.timestamps.append(time.perf_counter())


	def epochsdone(self):
		return len(self.epochlosses)

	def savehist(self,filename):
		self.checkpoint()
		bk.save({'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'timestamps':self.timestamps},filename)


	def minibatch_epoch(self,minibatchsize):
		
		X,Y=util.randperm(self.X,self.Y)
		samples=X.shape[0]
		minibatchlosses=[]

		for a in range(0,samples,minibatchsize):
			c=min(a+minibatchsize,samples)

			x=X[a:c]
			y=Y[a:c]

			grad,loss=self.lossgrad([self.Ws,self.bs],x,y)

			updates,self.state=self.opt.update(grad,self.state,[self.Ws,self.bs])
			[self.Ws,self.bs]=optax.apply_updates([self.Ws,self.bs],updates)

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

			grad,loss=self.lossgrad([self.Ws,self.bs],x,y)


			rloss=loss/self.nullloss
			minibatchlosses.append(rloss)
			minibatchparamgrads=addparamgrads(minibatchparamgrads,grad)

			completeness=1.0*c/samples; bk.printbar(completeness,msg='Epoch '+str(round(completeness*100))+'% complete',printval=False,style=10*bk.bar+'minibatches done',emptystyle=' ')

		updates,self.state=self.opt.update(minibatchparamgrads,self.state,[self.Ws,self.bs])
		[self.Ws,self.bs]=optax.apply_updates([self.Ws,self.bs],updates)

		print('Parameter update'+200*' ')
		bk.printbar(rloss,msg='training loss  ',style='\u2592',hold=False)
		self.epochlosses.append(jnp.average(minibatchlosses))


def addparamgrads(G1,G2):
	if G1==None:
		return G2
	else:
		return [[a1+a2 for a1,a2 in zip(g1,g2)] for g1,g2 in zip(G1,G2)]
		




class TrainerWithValidation(Trainer):

	def __init__(self,widths,X__,Y__,fractionforvalidation=.1,batchmode='minibatch'):
		trainingsamples=round(X__.shape[0]*(1-fractionforvalidation))
		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]

		super().__init__(widths,X_train,Y_train,batchmode=batchmode)
		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		self.valerrorhist=[self.validationerror(loud=False)]

	def validationerror(self,loud=False):
		vallosses=self.individuallosses([self.Ws,self.bs],self.X_val,self.Y_val)/self.nullloss
		if loud:
			bk.printbar(jnp.average(vallosses),msg='validation loss',hold=False)
		return vallosses

	def checkpoint(self):
		super().checkpoint()
		self.valerrorhist.append(self.validationerror(loud=True))

	def savehist(self,filename):
		bk.save({'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'valerrorhist':self.valerrorhist},filename)


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
def initandtrain(data_in_path,data_out_path,widths,batchsize,batchmode,action_on_pause=donothing): 
	X,Y=bk.get(data_in_path)
	T=TrainerWithValidation(widths,X,Y,batchmode=batchmode)
	try:
		while True:
			try:
				print('\nEpoch '+str(T.epochsdone()))
				T.epoch(batchsize)
			except KeyboardInterrupt:
				action_on_pause(T)
				continue
			T.checkpoint()
			T.savehist(data_out_path)
	except KeyboardInterrupt:
		print('\nEnding.\n')



def genW(k0,n,d,widths):

	if type(widths)!=list:
		print('Casting width to singleton list')
		widths=[widths]

	k1,*Wkeys=rnd.split(k0,100)
	k2,*bkeys=rnd.split(k0,100)

	Ws=[rnd.normal(k1,(widths[0],n,d))/math.sqrt(n*d)]
	for m1,m2,key in zip(widths,widths[1:]+[1],Wkeys):
		Ws.append(rnd.normal(key,(m2,m1))/math.sqrt(m1))

	bs=[rnd.normal(key,(m,)) for m,key in zip(widths,bkeys)]
	return Ws,bs


def AS_from_hist(path):
	Ws,bs=bk.get(path)['paramshist'][-1]
	@jax.jit
	def Af(X):
		return AS_tools.AS_NN(Ws,bs,X)
	return Af

def losses_from_hist(path):
	trainerrors=bk.get(path)['trainerrorhist']
	valerrors=[jnp.average(_) for _ in bk.get(path)['valerrorhist']]
	return trainerrors,valerrors
	



#----------------------------------------------------------------------------------------------------
# losses and gradients
#----------------------------------------------------------------------------------------------------




individualloss=util.sqlossindividual
collectiveloss=util.sqloss


@jax.jit
def individuallossesAS(Wb,X,Y):
	Ws,bs=Wb
	Z=AS_tools.AS_NN(Ws,bs,X)
	return individualloss(Y,Z)

@jax.jit
def collectivelossAS(Wb,X,Y):
	return jnp.average(individuallossesAS(Wb,X,Y))

@jax.jit
def lossgradAS(Wb,X,Y):
	Ws,bs=Wb
	loss,grad=jax.value_and_grad(collectivelossAS)([Ws,bs],X,Y)
	return grad,loss



@jax.jit
def individuallossesNS(Wb,X,Y):
	W,b=Wb
	Z=AS_tools.NN(W,b,X)
	return individualloss(Y,Z)

@jax.jit
def collectivelossNS(Wb,X,Y):
	return jnp.average(individuallossesNS(Wb,X,Y))

@jax.jit
def lossgradNS(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(collectivelossNS)([W,b],X,Y)
	return grad,loss


