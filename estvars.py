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
import permutations as perms
import partialsum as ps
import GPU_sum




def estvars(n,d,samples,key):
	instances=samples
	key1,key2=jax.random.split(key)
	W=jax.random.normal(key1,(instances,n,d))/jnp.sqrt(n*d)
	X=jax.random.normal(key2,(samples,n,d))
	outputs=[GPU_sum.sum_all_perms(W[i],X[i],'ReLU') for i in range(samples)]	
	bk.savedata(outputs,'batchdata/n='+str(n)+' | '+str(samples)+' samples | key='+str(key))
	return outputs



if __name__=='__main__':
	d=3
	rounds=1000
	nmax=int(sys.argv[1])
	seed=int(sys.argv[2])
	key=jax.random.PRNGKey(seed)
	key0,*keys=jax.random.split(key,rounds+2)
	samplenumbers={n:2**(nmax-n) for n in range(nmax+1)}
	log.log('sample numbers '+str(samplenumbers),loud=True)

	for i in range(rounds):
		_,*roundkeys=jax.random.split(keys[i],20)
		print('round '+str(i)+' '+100*'-')

		for n in range(3,nmax+1):
			estvars(n,d,samplenumbers[n],roundkeys[n])
			print('n='+str(n))
	
