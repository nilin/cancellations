import jax
import jax.numpy as jnp
import numpy as np
from ..functions import multivariate as mv
import itertools
from utilities import math as mathutil








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

def gen_hermitegaussproducts(n,d,envelopevariance=1):
	hermiteprods=util.fixparams(polynomial_products,hermite_nd_params(n,d))	
	envelope=lambda x:jnp.exp(-jnp.sum(jnp.square(x),axis=(-1))/(2*envelopevariance))
	return jax.jit(lambda X:envelope(X)[:,None]*hermiteprods(X))

#----------------------------------------------------------------------------------------------------
# test
#----------------------------------------------------------------------------------------------------



def test():
	for p in H_coefficients_list: print(p)

	





