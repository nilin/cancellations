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
import parallelpartialsum as p_sums
import permutations

def pickparams(N):
	if N<5040:
		return {'tasks':1,'smallblock':N,'bigblock':N}
	else:
		return {'tasks':8,'smallblock':630,'bigblock':5040}

if __name__=='__main__':

	ac_name=sys.argv[1]
	Wtype={'s':'separated','n':'normal','ss':'separated small','ns':'normal small'}[sys.argv[2]]

	data=bk.getdata(Wtype+'/WX')
	Ws=data['Ws']
	Xs=data['Xs']

	n=2
	while True:
		print(n)
		N=math.factorial(n)
		prefix='partialsums/'+Wtype+'/'+ac_name+' n='+str(n)+' range='
		if os.path.exists('data/'+prefix+'0 '+str(N)):
			pass#break
		else:
			p_sums.parallel_sum(Ws[n],Xs[n],ac_name,0,N,prefix,pickparams(N))
		n=n+1
