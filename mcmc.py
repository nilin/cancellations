# nilin


import jax
import jax.numpy as jnp
import jax.random as rnd
import time
import config as cfg


scaleby=jax.vmap(jnp.multiply,in_axes=(0,0))



class Sampler:
	
	def __init__(self,p,proposal,X0,seed=0,burntime=0):
		self.p=p
		self.runners=X0.shape[0]
		self.initkeys(seed)	
		self.X=X0
		self.proposal=proposal
		self.hist=[]
		print('burning')
		self.run(burntime)


	def step(self):
		X0=self.X
		X1=self.proposals(X0)

		ratios=self.p(X1)/self.p(X0)

		u=rnd.uniform(self.nextkey(),ratios.shape)
		accepted=ratios>u
		rejected=1-accepted
		self.X=scaleby(rejected,X0)+scaleby(accepted,X1)
	
	
	def proposals(self,X):
		keys=self.nextkeys(self.runners)
		return jax.vmap(self.proposal,in_axes=(0,0))(keys,X)


	def checkpoint(self):
		print('checkpoint')
		self.hist.append(self.X)
		
	def run(self,duration):
		t=time.perf_counter()
		while(time.perf_counter()<t+duration):
			self.step()
		self.checkpoint()
		
		
##----------------------------------------------------------------------------------------------------

	def nextkey(self):
		return jnp.squeeze(self.nextkeys(1))

	def nextkeys(self,n):	
		if n>len(self.keys):
			self.updatekeys()
		return jnp.array([self.keys.pop() for i in range(n)])

	def initkeys(self,seed):
		_,*self.keys=rnd.split(rnd.PRNGKey(seed),10**5)

	def updatekeys(self):
		_,*self.keys=rnd.split(self.keys[-1],10**5)



#----------------------------------------------------------------------------------------------------#

def test():

	#f=lambda x:jnp.exp(-(x-.5)**2)
	f=lambda x:x*(x<1)

	def proposal(key,x):
		return x+rnd.normal(key)*.1

	X0=rnd.uniform(rnd.PRNGKey(0),(1000,))
	sampler=Sampler(f,proposal,X0,burntime=1)


	for i in range(30):
		sampler.run(.1)

	X=jnp.concatenate(sampler.hist,axis=0)

	import seaborn as sns
	import matplotlib.pyplot as plt
	sns.kdeplot(X,bw=.1)
	plt.show()

if __name__=='__main__':
	test()
