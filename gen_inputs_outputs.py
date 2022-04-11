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




d=3
N=100000

def generate(*args):
	bk.log('\n'+str(jax.devices()[0])+'\n',loud=True)
	ac_name=args[0]
	nmin=int(args[1])
	nmax=int(args[2])
	seed=int(args[3])
	

	inputs=loadinputs(seed,nmin,nmax)
	Ws,Xs=inputs['Ws'],inputs['Xs']
	samples_done=loadtracker(seed,nmin,nmax,ac_name)

	samples_per_round={n:round(2*4**(nmax-n)) for n in range(nmin,nmax+1)}

	while samples_done[nmax]<N:

		for n in range(nmin,nmax+1):

				
			endblock=min(N,samples_done[n]+samples_per_round[n])
			if endblock==samples_done[n]:
				continue

			W=Ws[n][samples_done[n]:endblock]
			X=Xs[n][samples_done[n]:endblock]

			bk.log(3*'\n'+'\nn='+str(n)+'\n'+str(endblock-samples_done[n])+' samples\n'+150*'=')

			outputs=GPU_sum.sum_perms(W,X,ac_name)

			samples_done[n]=endblock
			updatetracker(seed,nmin,nmax,ac_name,samples_done)
			updateoutputs(seed,nmin,nmax,ac_name,n,outputs)
	

def loadinputs(seed,nmin,nmax):
	path=genpath(seed,nmin,nmax)+'/inputs'
	if not os.path.exists(path):
		geninputs(seed,nmin,nmax,path)
	return bk.get(path)
	
def geninputs(seed,nmin,nmax,path):
	key0=jax.random.PRNGKey(seed)
	key1,key2=jax.random.split(key0)
	_,*Wkeys=jax.random.split(key1,100)
	_,*Xkeys=jax.random.split(key2,100)
	Ws={n:jax.random.normal(Wkeys[n],(N,n,d))/jnp.sqrt(n*d) for n in range(nmin,nmax+1)}
	Xs={n:jax.random.normal(Xkeys[n],(N,n,d)) for n in range(nmin,nmax+1)}
	bk.save({'Ws':Ws,'Xs':Xs,'Wkeys':Wkeys,'Xkeys':Xkeys},path)

def loadtracker(seed,nmin,nmax,ac_name):
	path=genpath(seed,nmin,nmax)+'/'+ac_name+' tracker'
	if not os.path.exists(path):
		#bk.save({'samples done':{n:0 for n in range(nmin,nmax+1)}},path)
		updatetracker(seed,nmin,nmax,ac_name,{n:0 for n in range(nmin,nmax+1)})
		prepoutputs(seed,nmin,nmax,ac_name)
	with open(path,'r') as f:
		return {int(ns.split()[0]):int(ns.split()[1]) for ns in f.readlines()}	

#def updatetracker(seed,nmin,nmax,ac_name,samples_done):
#	bk.save({'samples done':samples_done},genpath(seed,nmin,nmax)+'/'+ac_name+' tracker')

def updatetracker(seed,nmin,nmax,ac_name,samples_done):
	path=genpath(seed,nmin,nmax)+'/'+ac_name+' tracker'
	bk.makedirs(path)
	with open(path,'w') as f:
		f.write('\n'.join([str(n)+' '+str(s) for n,s in samples_done.items()]))
	

def updateoutputs(seed,nmin,nmax,ac_name,n,new_outputs):
	path=genpath(seed,nmin,nmax)+'/'+ac_name+' '+str(n)
	data=bk.get(path)
	outputs=jnp.concatenate([data['outputs'],new_outputs],axis=0)
	data['outputs']=outputs
	bk.save(data,path)
	
def prepoutputs(seed,nmin,nmax,ac_name):
	for n in range(nmin,nmax+1):
		path=genpath(seed,nmin,nmax)+'/'+ac_name+' '+str(n)
		bk.save({'outputs':jnp.array([])},path)

def genpath(seed,nmin,nmax):
	return 'data/range='+str(nmin)+' '+str(nmax)+' seed='+str(seed)
	



if __name__=='__main__':
	generate(*sys.argv[1:])
