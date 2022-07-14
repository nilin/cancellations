import GPU_sum
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
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

	def epochNS(self,minibatchsize):
		
		#X_train,Z_train=randperm(self.X_train,self.Z_train)
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


	def checkpoint(self):
		self.paramshistory.append((self.W,self.b))
		return jnp.average(self.epochlosses[-1])


	def savehist(self,filename):
		bk.save(self.paramshistory,filename)
		
				
		

def initandtrain(d,n,m,samples,batchsize,traintime,trainmode='AS'):
	T=Trainer(d,n,m,samples)

	variables={'d':d,'n':n,'m':m,'s':samples,'bs':batchsize}
	

	t0=time.perf_counter()
	loss='null'

	while time.perf_counter()<t0+traintime and (loss=='null' or loss>.005):
		if trainmode=='NS':
			T.epochNS(batchsize)
		if trainmode=='AS':
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
	m=100
	samples=1000 if trainmode=='AS' else 10**6
	batchsize=100 if trainmode=='AS' else 10000

	for d in [1,3]:
		print('d='+str(d))
		for n in range(1,8):
			print('n='+str(n))
			initandtrain(d,n,m,samples,batchsize,traintime,trainmode)
			print('\n')



	








