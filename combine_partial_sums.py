import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
import seaborn as sns
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import optax
import util
import sys
import os
import shutil
import cancellation as canc
import proxies
import parallelpartialsum as p_sums
import permutations




def combine(ac_name,Wtype,n):

	activation=util.activations[ac_name]
	dirpath='partialsums/'+Wtype
	prefix=dirpath+'/'+ac_name+' n='+str(n)+' range='

	a=0
	b=0
	N=math.factorial(n)

	checkpoints=[]
	blocksums=[]
	print('combining files:')

	step=min(N,120)
	while b<N:
		b=b+step

		filepath=prefix+str(a)+' '+str(b)
		if os.path.exists('data/'+filepath):
			print(filepath)
			data=bk.getdata(filepath)
			S=data['result']
			blocksums.append(S)
			checkpoints.append(b)
			a=b	
	print(b)
	assert(b==N)
	print('checkpoints: '+str(checkpoints))
	return sum(blocksums)/jnp.sqrt(N)


def test_combine(n):
	
	norm_=util.L2norm(combine('ReLU','normal',n))
	n_,norms=bk.getdata('normal/'+str(n)+'/ReLU')

	print(100*'-')	
	print(norm_)
	print(norms)
	assert(jnp.abs(jnp.log(norm_/norms[-1]))<.001)
	print(100*'-')	



"""
combine_partial_sums.py ReLU n 10
"""


ac_name=sys.argv[1]
Wtype={'s':'separated','n':'normal','ns':'normal small','ss':'separated small'}[input('W type: (n)normal, (s)eparated, ns=(n)ormall (s)mall, (s)eparated (s)mall :')]
nmin,nmax=[int(inp) for inp in input('range n_min n_max inclusive :').split()]

n_=list(range(nmin,nmax+1))
sums=[combine(ac_name,Wtype,n) for n in n_]

bk.savedata({'range':n_,'data':sums,'norms':[util.L2norm(Y) for Y in sums]},Wtype+'/'+ac_name)
