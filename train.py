import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import optax
import math
import universality
import sys
import matplotlib.pyplot as plt
#from plotuniversal import plot as plot3
import numpy as np
import scipy.stats as st
import time
import pdb



def distinguishable(x,y):
	u,p=st.mannwhitneyu(x,y,alternative='greater')

	return p<.25

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


####################################################################################################
	
class Trainer:
	def __init__(self,fn,samples,m,mode):

		self.n,self.d=self.set_traindata(fn,samples)
		self.m=m

		k0=rnd.PRNGKey(0)
		self.W,self.b=universality.genW(k0,self.n,self.d,m)

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
		self.lossgrad=universality.lossgradAS if mode=='AS' else univerality.lossgradNS


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

			rloss=loss/universality.lossfn(Y,0)
			losses.append(rloss)
			bk.printbar(rloss,'{:.4f}'.format(rloss))

		self.epochlosses.append(jnp.array(losses))

	def stale(self):
		l=len(self.epochlosses)

		if l<4:
			return False

		x=jnp.concatenate(self.epochlosses[int(l/2):int(l*3/4)])
		y=jnp.concatenate(self.epochlosses[int(l*3/4):])

		return x.size>50 and not distinguishable(x,y)

	



#class NS_trainer(Trainer):
#
#	def lossgrad(self,Wb,X,Y):
#		return universality.lossgradNS(Wb,X,Y)	
#
#
#class AS_Trainer(Trainer):	
#
#	def lossgrad(self,Wb,X,Y):
#		return universality.lossgradAS(Wb,X,Y)	


	
class ASNS_Trainer(Trainer):

	def __init__(self,fn,samples,m):
		super().__init__(fn,samples,m,'ASNS')
		self.enrichmentperms,self.enrichmentsigns=gen_swaps(self.n)


	def enrich_inputs(self,X,Y):
		X_=util.apply_on_n(self.enrichmentperms,X)
		Y_=self.enrichmentsigns[:,None]*jnp.squeeze(Y)[None,:]
		return util.flatten_first(X_),util.flatten_first(Y_)

#	def lossgrad(self,Wb,X,Y):
#		return universality.lossgradNS(Wb,X,Y)	




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

#	def lossgrad(self,Wb,X,Y):
#		return universality.lossgradNS(Wb,X,Y)	




		

def initandtrain(d,n,m,samples,batchsize,traintime,trainmode='AS'):
	T=ASNS_Trainer(d,n,m,samples)

	variables={'d':d,'n':n,'m':m,'s':samples,'bs':batchsize}
	

	t0=time.perf_counter()

	try:
		while time.perf_counter()<t0+traintime:
			T.epoch(batchsize)
			T.checkpoint()

			T.savehist('data/hists/'+trainmode+'_'+formatvars_(variables))

			if T.stale():
				break
	except KeyboardInterrupt:
		pass





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



	








