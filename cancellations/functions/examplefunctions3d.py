import jax
import jax.numpy as jnp
import numpy as np
from ..functions import multivariate as mv
from .. import functions
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

def hermite_nd_params(n,d):		
	return [[H_coefficients_list[p] for p in phi] for phi in gen_n_dtuples(n,d)]	


#----------------------------------------------------------------------------------------------------
# test
#----------------------------------------------------------------------------------------------------

psis=[examplefunctions.psi(i) for i in range(10)]
_Psis_=[[] for i in range(4)]

for d in [1,2,3]:

	tuples=gen_n_dtuples(6,d)
	for ijk in tuples:

		def psi(X,ijk=ijk):
			out=1
			for k,l in zip(ijk,range(d)):
				out*=psis[k](X[:,l])
			return out

			#by_dim=[psis[k](X[:,dim]) for k,dim in zip(ijk,range(d))]
			#return jnp.product(jnp.stack(by_dim,axis=-1),axis=-1)

#		import pdb
#		pdb.set_trace()

		_Psis_[d].append(psi)


	for i,psi in enumerate(_Psis_[d]):
		globals()['psi{}_{}d'.format(i,d)]=psi




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

	





