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
import permutations as perms
import GPU_sum




def sample_inputs_and_outputs(ac_name,n,d,instances,key,seed):
	bk.log(3*'\n'+'\nn='+str(n)+'\n'+str(instances)+' instances\n'+150*'=')
	W=jax.random.normal(key,(instances,n,d))/jnp.sqrt(n*d)
	outputs=GPU_sum.sum_perms(W,W,'gamma_'+ac_name)
	bk.savedata({'W':W,'outputs':jnp.array(outputs)},'seed='+str(seed)+'/gamma '+ac_name+' | n='+str(n)+' | '+str(instances)+' instances | key='+str(key))
	return outputs



def generate(*args):
	bk.log('\n'+str(jax.devices()[0])+'\n',loud=True)
	d=3
	rounds=1

	ac_name='ReLU'
	nmin=2
	nmax=12
	seed=0

#	ac_name=args[0]
#	nmin=int(args[1])
#	nmax=int(args[2])
#	seed=int(args[3])
	key=jax.random.PRNGKey(seed)
	key0,*keys=jax.random.split(key,rounds+2)
	r=2
	samplenumbers={n:round(min(10.0*2**(nmax-n),1000)) for n in range(nmin,nmax+1)}
	bk.log('sample numbers each round '+str(samplenumbers),loud=True)

	for i in range(rounds):
		_,*roundkeys=jax.random.split(keys[i],20)
		bk.log('round '+str(i)+' '+100*'-')

		for n in range(nmin,nmax+1):
			bk.log('n='+str(n))
			sample_inputs_and_outputs(ac_name,n,d,samplenumbers[n],roundkeys[n],seed)
	

if __name__=='__main__':
	generate(*sys.argv[1:])
