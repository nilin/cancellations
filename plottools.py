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

	def plot(self,axs,learned):
		for ax,x,y in zip(axs,self.lines,self.ys):
			ax.plot(self.interval,y,'b',label='target')
			ax.plot(self.interval,learned(x),'r',label='learned')
			ax.legend()
		



class Plotter(cfg.State):
	
	def registerstate(self,t,state):
		data=processstate(state)
		for k,val in data.items():
			self.remember(k,val,t=t)


	def processhist(self,paramshist,schedule):
		sc=cfg.Scheduler()
		timestamps,states=sc.filter(paramshist['timestamps'],paramshist['vals'])
		for t,state in zip(timestamps,states):
			registerstate(t,state)
		

	
		
