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





#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------


	
class Trainer:
	def __init__(self,mode,widths,X,Y):

		self.n,self.d=X.shape[-2:]
		self.X,self.Y=X,Y

		k0=rnd.PRNGKey(0)
		self.Ws,self.bs=genW(k0,self.n,self.d,widths)

		self.opt=optax.adamw(.01)
		self.state=self.opt.init((self.Ws,self.bs))

		self.epochlosses=[]
		self.paramshist=[]
		self.trainerrorhist=[]

		self.set_trainmode(mode)
		self.nullloss=collectiveloss(Y,0)


	def set_trainmode(self,mode):
		self.lossgrad=lossgradAS if mode=='AS' else lossgradNS
		self.individuallosses=individuallossesAS if mode=='AS' else individuallossesNS


	def checkpoint(self):
		self.paramshist.append((self.Ws,self.bs))
		self.trainerrorhist.append(jnp.average(self.epochlosses[-1]))


	def savehist(self,filename):
		bk.save({'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist},filename)


	def epoch(self,minibatchsize):
		
		X,Y=util.randperm(self.X,self.Y)
		samples=X.shape[0]
		losses=[]

		for a in range(0,samples,minibatchsize):
			c=min(a+minibatchsize,samples)

			x=X[a:c]
			y=Y[a:c]

			grad,loss=self.lossgrad((self.Ws,self.bs),x,y)

			updates,self.state=self.opt.update(grad,self.state,(self.Ws,self.bs))
			(self.Ws,self.bs)=optax.apply_updates((self.Ws,self.bs),updates)

			rloss=loss/self.nullloss
			losses.append(rloss)
			bk.printbar(rloss,'{:.4f}'.format(rloss))

		self.epochlosses.append(jnp.array(losses))





class TrainerWithValidation(Trainer):

	def __init__(self,mode,widths,X__,Y__,fractionforvalidation=.1):
		trainingsamples=round(X__.shape[0]*(1-fractionforvalidation))
		X_train,Y_train=X__[:trainingsamples],Y__[:trainingsamples]

		super().__init__(mode,widths,X_train,Y_train)
		self.X_val,self.Y_val=X__[trainingsamples:],Y__[trainingsamples:]
		self.valerrorhist=[self.validationerror()]

	def validationerror(self):
		return self.individuallosses((self.Ws,self.bs),self.X_val,self.Y_val)/self.nullloss

	def checkpoint(self):
		super().checkpoint()
		self.valerrorhist.append(self.validationerror())

	def savehist(self,filename):
		bk.save({'paramshist':self.paramshist,'trainerrorhist':self.trainerrorhist,'valerrorhist':self.valerrorhist},filename)


	def makingprogress(self,p_val=.10):
		return True if len(self.valerrorhist)<2 else distinguishable(self.valerrorhist[-2],self.valerrorhist[-1],p_val)

	def stale(self,p_val=.10):
		return not makingprogress(p_val)



	
def distinguishable(x,y,p_val=.10):
	u,p=st.mannwhitneyu(x,y,alternative='greater')
	return p<p_val




#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------

"""
press Ctrl-C to stop training
stopwhenstale either False (no stop) or p-value (smaller means earlier stopping)
"""
def initandtrain(data_in_path,data_out_path,mode,widths,batchsize,traintime=600,stopwhenstale=False): 

	X,Y=bk.get(data_in_path)
	T=TrainerWithValidation(mode,widths,X,Y)
	t0=time.perf_counter()
	try:
		while time.perf_counter()<t0+traintime:
			print('\nEpoch '+str(len(T.epochlosses)))
			T.epoch(batchsize)
			T.checkpoint()
			T.savehist(data_out_path)

			if type(stopwhenstale)==float and T.stale(stopwhenstale):
				print('stale, stopping')
				break
	except KeyboardInterrupt:
		pass


def genW(k0,n,d,widths):
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
	W,b=Wb
	Z=AS_tools.AS_NN(W,b,X)
	return individualloss(Y,Z)

@jax.jit
def collectivelossAS(Wb,X,Y):
	return jnp.average(individuallossesAS(Wb,X,Y))

@jax.jit
def lossgradAS(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(collectivelossAS)((W,b),X,Y)
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
	loss,grad=jax.value_and_grad(collectivelossNS)((W,b),X,Y)
	return grad,loss


