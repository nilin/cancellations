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





def generate(*args):
	bk.log('\n'+str(jax.devices()[0])+'\n',loud=True)
	d=3
	ac_name=args[0]
	nmin=int(args[1])
	nmax=int(args[2])
	seed=int(args[3])
	
	key0=jax.random.PRNGKey(seed)
	key1,key2=jax.random.split(key0)
	#_,*Wkeys=jax.random.split(key1,100)
	_,*Xkeys=jax.random.split(key2,100)


	N=100000

	Ws=bk.getdata('w_packing')['ws']

	Ws={n:jnp.repeat(Ws[n],N//Ws[n].shape[0],axis=0) for n in range(nmin,nmax+1)}
	#Ws={n:jax.random.normal(Wkeys[n],(N,n,d))/jnp.sqrt(n*d) for n in range(nmin,nmax+1)}
	Xs={n:jax.random.normal(Xkeys[n],(N,n,d)) for n in range(nmin,nmax+1)}
	outputs={n:jnp.array([]) for n in range(nmin,nmax+1)}

	samples_per_round={n:round(2*4**(nmax-n)) for n in range(nmin,nmax+1)}
	samples_done={n:0 for n in range(nmin,nmax+1)}

	while samples_done[nmax]<N:

		for n in range(nmin,nmax+1):
				
			endblock=min(N,samples_done[n]+samples_per_round[n])
			if endblock==samples_done[n]:
				continue

			W=Ws[n][samples_done[n]:endblock]
			X=Xs[n][samples_done[n]:endblock]

			bk.log(3*'\n'+'\nn='+str(n)+'\n'+str(endblock-samples_done[n])+' samples\n'+150*'=')

			output=GPU_sum.sum_perms(W,X,ac_name)
			outputs[n]=jnp.concatenate([outputs[n],output])

			if samples_done[n]>0:			
				try:
					os.remove('data/packing seed='+str(seed)+'/'+ac_name+' | n='+str(n)+' | '+str(samples_done[n])+' samples')
				except:
					pass
			bk.savedata({'W':Ws[n][:endblock],'X':Xs[n][:endblock],'outputs':outputs[n],'seed':seed},'packing seed='+str(seed)+'/'+ac_name+' | n='+str(n)+' | '+str(endblock)+' samples')

			samples_done[n]=endblock
	
	

if __name__=='__main__':
	#generate(*sys.argv[1:])
	generate('ReLU',2,12,123)
