# nilin


import jax
import jax.numpy as jnp
import jax.random as rnd
import re
import os
import math
from . import numutil
from collections import deque

from cancellations.utilities import sysutil

from ..utilities import tracking


scaleby=jax.vmap(jnp.multiply,in_axes=(0,0))


class DynamicSampler:
	
	def __init__(self,_p_,proposalfn,X0):
		self._p_=_p_
		self.runners=X0.shape[0]
		self.X=X0
		self.proposalfn=proposalfn
		self.hist=[]
		
	def step(self,p_params):
		X0=self.X
		X1=self.proposals(X0)
		ratios=self._p_(p_params,X1)/self._p_(p_params,X0)
		u=rnd.uniform(tracking.nextkey(),ratios.shape)
		accepted=ratios>u
		rejected=1-accepted
		self.X=scaleby(rejected,X0)+scaleby(accepted,X1)
		return self.X
	
	def proposals(self,X):
		return self.proposalfn(tracking.nextkey(),X)

class Sampler(DynamicSampler):
	
	def __init__(self,p,proposalfn,X0):
		_p_=numutil.dummyparams(p)
		super().__init__(_p_,proposalfn,X0)
		
	def step(self):
		return super().step(None)

def gaussianstepproposal(var):
    return lambda key,X: X+rnd.normal(key,X.shape)*math.sqrt(var)




class SamplesPipe(Sampler):

	def __init__(self,X,*Ys,minibatchsize):
		self.X=X
		self.Ys=Ys
		self.minibatches=deque([])
		self.minibatchsize=minibatchsize

	def step(self):
		if len(self.minibatches)==0:
			self.prepnextepoch()
		return self.minibatches.popleft()

	def prepnextepoch(self,permute=True):
		if permute: self.X,*self.Ys=numutil.randperm(self.X,*self.Ys)
		self.minibatches=deque(numutil.chop(self.X,*self.Ys,blocksize=self.minibatchsize))





class InputExhausted(Exception): pass

class LoadSamplesPipe(Sampler):

	def __init__(self,path,burnfraction=.25,cycle=False):
		self.i=0
		self.cycle=cycle

		pattern=re.compile('block (.*)-(.*)')

		filenames=[fn for fn in os.listdir(path) if pattern.match(fn) is not None]
		intervals=[tuple(int(a) for a in pattern.match(fn).groups()) for fn in filenames]

		intervals,filenames=zip(*sorted(zip(intervals,filenames)))

		self.blocksize=int(intervals[0][1])-int(intervals[0][0])
		self.n_intervals=round(len(intervals)*(1-burnfraction))

		self.filepaths=deque([path+fn for fn in filenames[-self.n_intervals:]])

	def step(self):
		if self.i%self.blocksize==0:
			if len(self.filepaths)==0: raise InputExhausted('')
			else: filepath=self.filepaths.popleft()
			if self.cycle: self.filepaths.append(filepath)

			self.currentblock=sysutil.load(filepath)

		self.X=self.currentblock[self.i%self.blocksize]
		self.i+=1
		return self.X





def bootstrap_confinterval(samples,nresamples=100,q=jnp.array([.05,.95])):
	(N,)=samples.shape
	resampledaverages=jnp.average(rnd.choice(tracking.nextkey(),samples,(nresamples,N)),axis=-1)
	return jnp.quantile(resampledaverages,q)







#
#
#class PotentialExpectation:
#
#	def __init__(self,O,X,p0):
#		self.X=X
#		self.O_X=O(X)
#		self.p0_X=p0(X)
#
#	# expectation of O(X) under X~p.
#	@jax.jit
#	def E(self,p):
#		ratio=self.p(X)/self.p0_X
#		return jnp.sum(ratio*self.O_X)/jnp.sum(ratio)
#
#
#
#
#
#def square(f,**kw):
#	return jax.jit(lambda X:f(X,**kw)**2)
#
#


#----------------------------------------------------------------------------------------------------#
#
#def test():
#
#	#f=lambda x:jnp.exp(-(x-.5)**2)
#	f=lambda x:x*(x<1)
#
#	def proposal(key,x):
#		return x+rnd.normal(key)*.1
#
#	X0=rnd.uniform(rnd.PRNGKey(0),(1000,))
#	sampler=Sampler(f,proposal,X0,burnsteps=50)
#
#
#	for i in range(50):
#		sampler.run(10)
#
#	X=jnp.concatenate(sampler.hist,axis=0)
#
#	import seaborn as sns
#	import matplotlib.pyplot as plt
#	sns.kdeplot(X,bw=.1)
#	plt.show()
#
#if __name__=='__main__':
#	test()
#