import numpy as np
import math
import pickle
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
import GPU_sum




def sample_inputs_and_outputs(ac_name,n,d,samples,key):
	instances=samples
	key1,key2=jax.random.split(key)
	W=jax.random.normal(key1,(instances,n,d))/jnp.sqrt(n*d)
	X=jax.random.normal(key2,(samples,n,d))
	outputs=[GPU_sum.sum_all_perms(W[i],X[i],ac_name) for i in range(samples)]	
	bk.savedata({'W':W,'X':X,'outputs':jnp.array(outputs)},ac_name+' | n='+str(n)+' | '+str(samples)+' samples | key='+str(key))
	return outputs



def generate(*args):
	log.log('\n'+str(jax.devices()[0])+'\n',loud=True)
	d=3
	rounds=1000
	ac_name=args[0]
	nmin=int(args[1])
	nmax=int(args[2])
	seed=int(args[3])
	key=jax.random.PRNGKey(seed)
	key0,*keys=jax.random.split(key,rounds+2)
	r=1.8
	samplenumbers={n:min(math.ceil(r**(nmax-n))*2,1000) for n in range(nmin,nmax+1)}
	log.log('sample numbers each round '+str(samplenumbers),loud=True)

	for i in range(rounds):
		_,*roundkeys=jax.random.split(keys[i],20)
		print('round '+str(i)+' '+100*'-')

		for n in range(nmin,nmax+1):
			sample_inputs_and_outputs(ac_name,n,d,samplenumbers[n],roundkeys[n])
			print('n='+str(n))
	

if __name__=='__main__':
	generate(*sys.argv[1:])
