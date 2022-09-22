import jax
import jax.numpy as jnp
import numpy as np
from ..functions import multivariate as mv
from ..functions import functions
import itertools
from . import examplefunctions
from ..utilities import numutil as mathutil, numutil




# not currently supported







#----------------------------------------------------------------------------------------------------
# for d>1
# f1,..,fn need only be pairwise different in one space dimension
#----------------------------------------------------------------------------------------------------

# S+k-1 choose k-1
def sumsto(k,S):
	return [[b-a-1 for a,b in zip((-1,)+t,t+(S+k-1,))] for t in itertools.combinations(range(S+k-1),k-1)]


def gen_n_dtuples(n,d):
	s=0
	out=[]
	while len(out)<n:
		out=out+sumsto(d,s)
		s+=1
		
	return out[:n]

def n_dtuples_maxdegree(n,d):
	return max([max(t) for t in gen_n_dtuples(n,d)])

#def hermite_nd_params(n,d):		
#	return [[H_coefficients_list[p] for p in phi] for phi in gen_n_dtuples(n,d)]	


#----------------------------------------------------------------------------------------------------
# test
#----------------------------------------------------------------------------------------------------


def genpsi(d,ijk):
	def psi(X):
		out=1
		for k,l in zip(ijk,range(d)):
			out*=examplefunctions.psi(k)(X[:,l])
		return out
	return psi

Psis=[[] for i in range(4)]

for d in [1,2,3]:
	for ijk in gen_n_dtuples(10,d):
		Psis[d].append(genpsi(d,ijk))

	for i,psi in enumerate(Psis[d]):
		fname='psi{}_{}d'.format(i,d)
		globals()[fname]=psi
		setattr(functions,fname,psi)
		#functions.definefunctions(fname,psi)



def test():
	print([k for k in globals().keys() if 'psi' in k])

	import matplotlib.pyplot as plt

	I=jnp.arange(-3,3,.02)



	X1,X2=jnp.meshgrid(I,I)
	X=jnp.stack([X1,X2],axis=-1)

	fig,axs=plt.subplots(5)
	for i,ax in enumerate(axs):
		Y=jax.vmap(globals()['psi{}_2d'.format(i)])(X)
		print(Y.shape)
		ax.pcolormesh(X1,X2,Y)
	plt.show()

	#for p in H_coefficients_list: print(p)

	




