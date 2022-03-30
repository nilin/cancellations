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
import gen_inputs_outputs as gen




ws=bk.getdata('w_packing')['ws']

bk.log('\n'+str(jax.devices()[0])+'\n',loud=True)
d=3
nmin=2#int(args[1])
nmax=12#int(args[2])
seed=0#int(args[3])

n_=range(nmin,nmax+1)


key=jax.random.PRNGKey(0)
key0,*keys=jax.random.split(key,100)
samples=10

gamma_output={n:GPU_sum.sum_perms(ws[n],ws[n],'gamma_ReLU') for n in n_}
normal_output={n:GPU_sum.sum_perms(jnp.repeat(ws[n],samples//ws[n].shape[0],axis=0),jax.random.normal(keys[n],(samples,n,d)),'ReLU') for n in n_}

bk.savedata({'range':n_,'gamma':gamma_output,'normal':normal_output},'w_packing ReLU')
	

