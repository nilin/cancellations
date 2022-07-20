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

		self.paramshistory=[]
		self.epochlosses=[]

		self.set_trainmode(mode)


	def set_trainmode(self,mode):
		self.lossgrad=lossgradAS if mode=='AS' else lossgradNS


	def checkpoint(self):
		self.paramshistory.append((self.Ws,self.bs))
		return jnp.average(self.epochlosses[-1])


	def savehist(self,filename):
		bk.save(self.paramshistory,filename)


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
def initandtrain(data_in_path,data_out_path,mode,widths,batchsize,traintime=600,stopwhenstale=.10): 

	X,Y=bk.get(data_in_path)
	T=Trainer(mode,widths,X,Y)
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
	hist=bk.get(path)
	Ws,bs=hist[-1]
	@jax.jit
	def Af(X):
		return AS_tools.AS_NN(Ws,bs,X)
	return Af


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

	pass
#	key=rnd.PRNGKey(0)
#	Ws,bs=genW(key,2,3,[10,9])
#
#	print([W.shape for W in Ws])
#	print()
#	print([b.shape for b in bs])


#	traintime=int(sys.argv[1])	
#	trainmode=sys.argv[2]
#	nmax=int(sys.argv[3])
#
#	m=100
#	samples=1000 if trainmode=='AS' else 10**6
#	batchsize=100 if trainmode=='AS' else 10000
#
#	for d in [1,3]:
#		print('d='+str(d))
#		for n in range(2,nmax+1):
#
#			
#
#			print('n='+str(n))
#			initandtrain(d,n,m,samples,batchsize,traintime,trainmode)
#			print('\n')



	








