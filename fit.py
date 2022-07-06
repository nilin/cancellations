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
from plotuniversal import plot as plot3
import numpy as np




def randperm(*args):
#	X=args[0]
#	n=X.shape[0]
#	p=np.random.permutation()
#	return [jnp.stack([Y[p_i] for p_i in p]) for Y in args]
	return args
	


class Trainer:
	def __init__(self,d,n,m,samples):
		self.d,self.n,self.m,self.samples=d,n,m,samples

		k0=rnd.PRNGKey(0)
		self.W,self.b=universality.genW(k0,n,d,m)

		X_train=bk.get('data/X_train_n='+str(n)+'_d='+str(d))
		Y_train=bk.get('data/Y_train_n='+str(n)+'_d='+str(d)+'_m=1')

		self.X_train=X_train[:samples]
		self.Y_train=Y_train[:samples]

		self.opt=optax.adamw(.01)
		self.state=self.opt.init((self.W,self.b))

		self.paramshistory=[]


	def epoch(self,minibatchsize):

		X_train,Y_train=randperm(self.X_train,self.Y_train)
	
		for a in range(0,self.samples,minibatchsize):
			c=min(a+minibatchsize,self.samples)

			X=X_train[a:c]
			Y=Y_train[a:c]

			grad,loss=universality.lossgrad((self.W,self.b),X,Y)

			updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
			(self.W,self.b)=optax.apply_updates((self.W,self.b),updates)

			rloss=loss/universality.lossfn(Y,0)
			bk.printbar(rloss,rloss)



	def checkpoint(self):
		self.paramshistory.append((self.W,self.b))


	def savehist(self,filename):
		bk.save(self.paramshistory,filename)
		
				
		


def testerror(Wb,samples=100):
	W,b=Wb
	m,n,d=W[0].shape
	X=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
	Y=bk.get('data/Y_test_n='+str(n)+'_d='+str(d)+'_m=1')

	X=X[:samples]
	Y=Y[:samples]

	return universality.batchloss(Wb,X,Y)/universality.lossfn(Y,0)
	

def ploterrorhist(variables):
	hist=bk.get('data/hists/'+bk.formatvars_(variables))
	errorhist=[testerror(Wb) for Wb in hist]
	plt.plot(errorhist)
	plt.show()
	




#	if d==1 and n==3:
#		plotting=True
#	else:
#		plotting=False
#
#	if plotting:
#		plt.ion()
#		fig=plt.figure()
#
#		ax0=fig.add_subplot(1,3,1,projection='3d')
#		ax1=fig.add_subplot(1,3,2,projection='3d')
#		ax2=fig.add_subplot(1,3,3,projection='3d')
#		X_plot=X_test[:10000]
#		Y_plot=Y_test[:10000]
#		plot3(X_plot,Y_plot,ax0)
#
#		_=input("press enter to start fitting")



	#epochs=100

		#if plotting:
		#
		#	Z_plot=universality.sumperms(W,b,X_plot)

		#	ax2.cla()
		#	plot3(X_plot,Z_plot,ax2)
		#	plt.draw()
		#	plt.pause(0.0001)


#			if plotting:
#				Z=universality.sumperms(W,b,X)
#
#				ax1.cla()
#				plot3(X,Z,ax1)
#				plt.draw()
#				plt.pause(0.0001)


def formatvars_(D):
	D_={k:v for k,v in D if k not in {'s','bs'}}
	return bk.formatvars(D_)


if __name__=="__main__":

#	print('\n\nargs d n m samples minibatchsize\n\n')
#
#	d=int(sys.argv[1])
#	n=int(sys.argv[2])
#	m=int(sys.argv[3])
#	samples=int(sys.argv[4])
#	minibatchsize=int(sys.argv[5])

	variables=bk.formatvars(sys.argv[1:])
	d,n,m,samples,minibatchsize=[variables[n] for n in ['d','n','m','s','bs']]

	mode='train'
	mode='plot'

	if mode=='train':
		T=Trainer(d,n,m,samples)
		epochs=100
		for e in range(epochs):
			T.epoch(minibatchsize)
			T.checkpoint()

			if e%10==0:
				T.savehist('data/hists/'+formatvars_(variables))
				#T.savehist('data/hists/'+str(sys.argv[1:]))
	if mode=='plot':
		ploterrorhist(variables)
		

	
	





