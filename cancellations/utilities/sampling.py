# nilin


import jax
import jax.numpy as jnp
import jax.random as rnd

from ..utilities import tracking


scaleby=jax.vmap(jnp.multiply,in_axes=(0,0))


class Sampler:
	
	def __init__(self,p,proposalfn,X0):
		self.p=p
		self.runners=X0.shape[0]
		self.X=X0
		self.proposalfn=proposalfn
		self.hist=[]
		
	def step(self):
		X0=self.X
		X1=self.proposals(X0)
		ratios=self.p(X1)/self.p(X0)
		u=rnd.uniform(tracking.nextkey(),ratios.shape)
		accepted=ratios>u
		rejected=1-accepted
		self.X=scaleby(rejected,X0)+scaleby(accepted,X1)
	
	def proposals(self,X):
		return self.proposalfn(tracking.nextkey(),X)





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