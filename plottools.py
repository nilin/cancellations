import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
import optax
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import learning
import pdb
from collections import deque




def samplepoints(X,Y,nsamples):
	p=Y**2
	p=p/jnp.sum(p)
	#I=jnp.random.choice(,range(len(p)),nsamples,p=p)
	I=rnd.choice(cfg.nextkey(),jnp.arange(len(p)),(nsamples,),p=p)
	return X[I]
	



def linethrough(x,interval):
	corner=np.zeros_like(x)
	corner[0][0]=1
	x_rest=(1-corner)*x
	X=interval[:,None,None]*corner[None,:,:]+x_rest[None,:,:]
	return X




class CrossSections:
	def __init__(self,X,Y,target,nsections,fineness=500):
		x0s=samplepoints(X,Y,nsections)
		self.interval=jnp.arange(-1,1,2/fineness)
		self.lines=[linethrough(x0,self.interval) for x0 in x0s]
		self.ys=[target(line) for line in self.lines]
		self.target=target

	def plot(self,axs,learned):
		for ax,x,y in zip(axs,self.lines,self.ys):
			ax.plot(self.interval,y,'b',label='target')
			ax.plot(self.interval,learned(x),'r',label='learned')
			ax.legend()
		
		
"""

#				
#def plotalongline(ax,learned,target,X,Y,**kwargs):
#
#
#	#x0=X[jnp.argmax(Y**2)]
#
#
#	cfg.debuglog(x0.shape)
#	I,x=linethrough(x0,**kwargs)
#
#	ax.plot(I,target(x),'b',label='target')
#	ax.plot(I,learned(x),'r',label='learned')
#	ax.legend()
#
#


#
#def partition(bins,x,*ys):
#	bin_nrs=np.digitize(x,bins)
#	blocks=[np.where(bin_nrs==b)[0] for b in range(bin_nrs[-1]+1)]
#	return [[np.array(y)[I] for I in blocks] for y in (x,)+ys]
#
#	
#
#
#def ploterrorhist(ax,hists,logscale=False):
#
#	train=hists['minibatch loss']
#	test=hists['test loss']
#	t_train_blocks,train_loss_blocks=partition(test['timestamps'],train['timestamps'],train['vals'])
#	t_train,train_loss=[np.average(t) for t in t_train_blocks],[np.average(l) for l in train_loss_blocks]
#	ax.plot(t_train,train_loss,'rd--',label='training loss')
#	ax.plot(test['timestamps'],test['vals'],'bo-',label='test loss')
#	
#	ax.legend()
#	ax.set_xlabel('seconds')
#	if logscale:
#		ax.set_yscale('log')
#	else:
#		ax.set_ylim(0,1)
#
#
#
"""
