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
	def __init__(self,data_in_path,mode,m,samples):

		self.n,self.d=self.set_traindata(data_in_path,samples)
		self.m=m

		k0=rnd.PRNGKey(0)
		self.W,self.b=genW(k0,self.n,self.d,m)

		self.opt=optax.adamw(.01)
		self.state=self.opt.init((self.W,self.b))

		self.paramshistory=[]
		self.epochlosses=[]

		self.set_trainmode(mode)


	def set_traindata(self,fn,samples):
		X_train,Y_train=bk.get(fn)
		self.X_train=X_train[:samples]
		self.Y_train=Y_train[:samples]
		self.samples=samples
		return X_train.shape[-2:]


	def set_trainmode(self,mode):
		self.lossgrad=lossgradAS if mode=='AS' else lossgradNS


	def checkpoint(self):
		self.paramshistory.append((self.W,self.b))
		return jnp.average(self.epochlosses[-1])


	def savehist(self,filename):
		bk.save(self.paramshistory,filename)


	def epoch(self,minibatchsize):
		
		X_train,Y_train=util.randperm(self.X_train,self.Y_train)

		losses=[]

		for a in range(0,self.samples,minibatchsize):
			c=min(a+minibatchsize,self.samples)

			X=X_train[a:c]
			Y=Y_train[a:c]

			grad,loss=self.lossgrad((self.W,self.b),X,Y)

			updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
			(self.W,self.b)=optax.apply_updates((self.W,self.b),updates)

			rloss=loss/lossfn(Y,0)
			losses.append(rloss)
			bk.printbar(rloss,'{:.4f}'.format(rloss))

		self.epochlosses.append(jnp.array(losses))

	def stale(self,p_val=.10):
		losses=jnp.concatenate(self.epochlosses)
		l=len(losses)
		if l<100:
			return False
		x=losses[round(l/3):round(l*2/3)]
		y=losses[round(l*2/3):]
		return not distinguishable(x,y,p_val)



	
def distinguishable(x,y,p_val=.10):
	u,p=st.mannwhitneyu(x,y)#,alternative='greater')
	return p<p_val




#----------------------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------------------

"""
press Ctrl-C to stop training
stopwhenstale either False (no stop) or p-value (smaller means earlier stopping)
"""
def initandtrain(data_in_path,data_out_path,mode,m,samples,batchsize,traintime=600,stopwhenstale=.10): 
	T=Trainer(data_in_path,mode,m,samples)
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


def genW(k0,n,d,m=10,randb=False):
	k1,k2,k3=rnd.split(k0,3)
	W0=rnd.normal(k1,(m,n,d))/math.sqrt(n*d)
	W1=rnd.normal(k2,(1,m))/math.sqrt(m)
	W=[W0,W1]
	b=[rnd.normal(k3,(m,))]
	if randb:
		b=[rnd.normal(k0,(m,))]
	return W,b




#----------------------------------------------------------------------------------------------------
# losses and gradients
#----------------------------------------------------------------------------------------------------




lossfn=util.sqloss


def batchlossAS(Wb,X,Y):
	W,b=Wb
	Z=AS_tools.AS_NN(W,b,X)
	return lossfn(Y,Z)


def lossgradAS(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(batchlossAS)((W,b),X,Y)
	return grad,loss



@jax.jit
def batchlossNS(Wb,X,Y):
	W,b=Wb
	Z=AS_tools.NN(W,b,X)
	return lossfn(Y,Z)

@jax.jit
def lossgradNS(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(batchlossNS)((W,b),X,Y)
	return grad,loss




#----------------------------------------------------------------------------------------------------
# other training procedures
#----------------------------------------------------------------------------------------------------

def gen_swaps(n,with_Id=True):

	I=list(range(n))
	swaps,signs=([I],[1]) if with_Id else ([],[])
	for i in range(n-1):
		for j in range(i+1,n):
			_=I.copy()
			_[i],_[j]=j,i
			swaps.append(_)
			signs.append(-1)
	return permutations.perm_as_matrix(swaps),jnp.array(signs)

class Randgen():
	def __init__(self,seed=0):
		self.key=rnd.PRNGKey(seed)

	def genint(self,k,nsamples=1):
		out=rnd.randint(self.key,(nsamples,),0,k)
		_,self.key=rnd.split(self.key)
		return jnp.squeeze(out)

class ASNS_Trainer(Trainer):

	def __init__(self,fn,samples,m):
		super().__init__(fn,samples,m,'ASNS')
		self.enrichmentperms,self.enrichmentsigns=gen_swaps(self.n)


	def enrich_inputs(self,X,Y):
		X_=util.apply_on_n(self.enrichmentperms,X)
		Y_=self.enrichmentsigns[:,None]*jnp.squeeze(Y)[None,:]
		return util.flatten_first(X_),util.flatten_first(Y_)

class ASNS_Trainer_2(Trainer):

	def __init__(self,fn,samples,m):
		super().__init__(fn,samples,m,'ASNS2')
		self.swaps,_=gen_swaps(self.n,False)
		self.randgen=Randgen()

	def randswap(self):
		swap_id=self.randgen.genint(len(self.swaps))
		return self.swaps[swap_id]
		

	def enrich_inputs(self,X,Y):
		swap=self.randswap()
		X_=jnp.concatenate([X,util.apply_on_n(swap,X)],axis=0)
		Y_=jnp.concatenate([Y,-Y])
		return X_,Y_

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


def formatvars_(D):
	D_={k:v for k,v in D.items() if k not in {'s','bs'}}
	return bk.formatvars_(D_)


if __name__=="__main__":


	traintime=int(sys.argv[1])	
	trainmode=sys.argv[2]
	nmax=int(sys.argv[3])

	m=100
	samples=1000 if trainmode=='AS' else 10**6
	batchsize=100 if trainmode=='AS' else 10000

	for d in [1,3]:
		print('d='+str(d))
		for n in range(2,nmax+1):

			

			print('n='+str(n))
			initandtrain(d,n,m,samples,batchsize,traintime,trainmode)
			print('\n')



	








