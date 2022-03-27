import numpy as np
import math
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import util
import sys
import os
import shutil
import random
import multiprocessing as mp
import log
import permutations
import partialsum as ps
import GPU_sum





"""
gen_partial_sum.py ReLU n 10
"""


class Est_tracker:
	def __init__(self,batches):
		self.batches=batches
		self.sums=batches*[0]
		self.sizes=batches*[0]
		self.i=0
	
	def update(self,value):
		self.sums[self.i%self.batches]=self.sums[self.i%self.batches]+value
		self.sizes[self.i%self.batches]=self.sizes[self.i%self.batches]+1
		self.i=self.i+1
		
	def est(self):
		ests=jnp.array(self.sums)/jnp.array(self.sizes)
		est=jnp.average(ests)
		std_of_ests=jnp.std(ests)
		CV=(std_of_ests/est)/jnp.sqrt(self.batches)
		return est,CV


if __name__=='__main__':

	n=int(sys.argv[1])
	d=3

	samples=100
	instances=samples

	seed=int(sys.argv[2])
	key=jax.random.PRNGKey(seed)
	key1,key2=jax.random.split(key)
	W=jax.random.normal(key1,(instances,n,d))/jnp.sqrt(n*d)
	X=jax.random.normal(key2,(samples,n,d))

	outputs=[]	
	meansquaretracker=Est_tracker(10)

	for i in range(samples):
		out=GPU_sum.sum_all_perms(W[i],X[i],'ReLU')
		outputs.append(out)
		meansquaretracker.update(out**2)
		est,CV=meansquaretracker.est()
		#bk.printbar(CV,'coefficient of variation for estimated variances')

	bk.savedata(outputs,'n='+str(n)+' | '+str(samples)+' samples | randseed='+str(seed))
		

