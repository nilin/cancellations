import GPU_sum
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import permutations
import GPU_sum
import optax
import math
import universality
import sys
import matplotlib.pyplot as plt
#from plotuniversal import plot as plot3
import numpy as np
import time




def randperm(*args):
	X=args[0]
	n=X.shape[0]
	p=np.random.permutation(n)
	return [jnp.stack([Y[p_i] for p_i in p]) for Y in args]
	return args
	

# apply matrix A[...,:,:] on X[...,:,.]
def apply_on_n(A,X):

	_=jnp.dot(A,X)
	out= jnp.swapaxes(_,len(A.shape)-2,-2)

	return out



def flatten_first(X):
	blocksize=X.shape[0]*X.shape[1]
	shape=X.shape[2:]
	return jnp.reshape(X,(blocksize,)+shape)
	



class Trainer:
	def __init__(self,d,n,m,samples):
		self.d,self.n,self.m,self.samples=d,n,m,samples

		k0=rnd.PRNGKey(0)
		self.W,self.b=universality.genW(k0,n,d,m)

		X_train=bk.get('data/X_train_n='+str(n)+'_d='+str(d))
		Y_train=bk.get('data/Y_train_n='+str(n)+'_d='+str(d)+'_m=1')
		Z_train=bk.get('data/Z_train_n='+str(n)+'_d='+str(d)+'_m=1')

		self.X_train=X_train[:samples]
		self.Y_train=Y_train[:samples]
		self.Z_train=Z_train[:samples]

		self.opt=optax.adamw(.01)
		self.state=self.opt.init((self.W,self.b))

		self.paramshistory=[]
		self.epochlosses=[]


	def checkpoint(self):
		self.paramshistory.append((self.W,self.b))
		return jnp.average(self.epochlosses[-1])


	def savehist(self,filename):
		bk.save(self.paramshistory,filename)
	


	
class NS_Trainer(Trainer):

	def epoch(self,minibatchsize):
		
		X_train,Z_train=self.X_train,self.Z_train
		losses=[]

		for a in range(0,self.samples,minibatchsize):
			c=min(a+minibatchsize,self.samples)

			X=X_train[a:c]
			Z=Z_train[a:c]

			grad,loss=universality.lossgradNS((self.W,self.b),X,Z)

			updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
			(self.W,self.b)=optax.apply_updates((self.W,self.b),updates)

			rloss=loss/universality.lossfnNS(Z,0)
			losses.append(rloss)
			bk.printbar(rloss,rloss)
		self.epochlosses.append(losses)


class AS_trainer(Trainer):	

	def epoch(self,minibatchsize):

		X_train,Y_train=randperm(self.X_train,self.Y_train)
		losses=[]
	
		for a in range(0,self.samples,minibatchsize):
			c=min(a+minibatchsize,self.samples)

			X=X_train[a:c]
			Y=Y_train[a:c]

			grad,loss=universality.lossgrad((self.W,self.b),X,Y)

			updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
			(self.W,self.b)=optax.apply_updates((self.W,self.b),updates)

			rloss=loss/universality.lossfn(Y,0)
			losses.append(rloss)
			bk.printbar(rloss,rloss)

		self.epochlosses.append(losses)


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
			
class ASNS_Trainer(Trainer):

	def __init__(self,d,n,m,samples):
		super().__init__(d,n,m,samples)
		self.enrichmentperms,self.enrichmentsigns=gen_swaps(n)


	def enrich_inputs(self,X,Y):
		X_=apply_on_n(self.enrichmentperms,X)
		Y_=self.enrichmentsigns[:,None]*jnp.squeeze(Y)[None,:]
		return flatten_first(X_),flatten_first(Y_)

	def epoch(self,minibatchsize):
		
		X_train,Y_train=self.X_train,self.Y_train
		losses=[]

		for a in range(0,self.samples,minibatchsize):
			c=min(a+minibatchsize,self.samples)

			X=X_train[a:c]
			Y=Y_train[a:c]

			X,Y=self.enrich_inputs(X,Y)

			grad,loss=universality.lossgradNS((self.W,self.b),X,Y)

			updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
			(self.W,self.b)=optax.apply_updates((self.W,self.b),updates)

			rloss=loss/universality.lossfn(Y,0)
			losses.append(rloss)
			bk.printbar(rloss,rloss)
		self.epochlosses.append(losses)

#
#class Randgen():
#	def __init__(self,seed=0):
#		self.key=rnd.PRNGKey(seed)
#
#	def genint(self,k,nsamples=1):
#		out=rnd.randint(self.key,(nsamples,),0,k)
#		_,self.key=rnd.split(self.key)
#		return jnp.squeeze(out)
#
#	
#
#class ASNS_Trainer(Trainer):
#
#	def __init__(self,d,n,m,samples):
#		super().__init__(d,n,m,samples)
#		self.swaps,_=gen_swaps(n,False)
#		self.randgen=Randgen()
#
#	def randswap(self):
#		swap_id=self.randgen.genint(len(self.swaps))
#		return self.swaps[swap_id]
#		
#
#	def enrich_inputs(self,X,Y):
#		swap=self.randswap()
#		X_=jnp.concatenate([X,apply_on_n(swap,X)],axis=0)
#		Y_=jnp.concatenate([Y,-Y])
#		return X_,Y_
#
#	def epoch(self,minibatchsize):
#		
#		X_train,Y_train=self.X_train,self.Y_train
#		losses=[]
#
#		for a in range(0,self.samples,minibatchsize):
#			c=min(a+minibatchsize,self.samples)
#
#			X=X_train[a:c]
#			Y=Y_train[a:c]
#
#			X,Y=self.enrich_inputs(X,Y)
#
#			grad,loss=universality.lossgradNS((self.W,self.b),X,Y)
#
#			updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
#			(self.W,self.b)=optax.apply_updates((self.W,self.b),updates)
#
#			rloss=loss/universality.lossfn(Y,0)
#			losses.append(rloss)
#			bk.printbar(rloss,rloss)
#		self.epochlosses.append(losses)
#


		

def initandtrain(d,n,m,samples,batchsize,traintime,trainmode='AS'):
	T=ASNS_Trainer(d,n,m,samples)

	variables={'d':d,'n':n,'m':m,'s':samples,'bs':batchsize}
	

	t0=time.perf_counter()
	loss='null'

	while time.perf_counter()<t0+traintime and (loss=='null' or loss>.005):
		T.epoch(batchsize)
		loss=T.checkpoint()

		#if e%10==0:
		T.savehist('data/hists/'+trainmode+'_'+formatvars_(variables))





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



	








