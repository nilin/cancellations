import jax
import jax.numpy as jnp
import numpy as np
from ..functions import multivariate as mv,functions
import itertools
from ..utilities import numutil as mathutil,textutil



#----------------------------------------------------------------------------------------------------
# polynomials
#----------------------------------------------------------------------------------------------------

def monomials(x,n):
	xk=jnp.ones_like(x)
	out=[]
	for k in range(n+1):
		out.append(xk)	
		xk=x*xk
	return jnp.stack(out,axis=-1)

def polynomial(coefficients,X):
	n=len(coefficients)-1
	return jnp.inner(monomials(X,n),coefficients)
		
	
#----------------------------------------------------------------------------------------------------
# Hermite polynomials
#----------------------------------------------------------------------------------------------------

			
def H_coefficients(n):
	if n==0:
		return [[1]]
	else:
		A=H_coefficients(n-1)
		a1=A[-1]+2*[0]
		a=[-a1[1]]
		for k in range(1,n+1):
			a.append(2*a1[k-1]-(k+1)*a1[k+1])
		A.append(a)
		return A


H_coefficients_list=[jnp.array(p) for p in H_coefficients(10)]


#----------------------------------------------------------------------------------------------------
# H_O_solution with -h-,m,k=1
#----------------------------------------------------------------------------------------------------


def psi(n):
	assert(n>0)
	p=mathutil.fixparams(polynomial,H_coefficients_list[n-1])
	psi_n=jax.jit(lambda x: jnp.exp(-x**2/2)*p(x))
	return psi_n

def totalenergy(n): return sum([i+1/2 for i in range(n)])



# load function definitions

# for i in range(1,11):
# 	fname='psi'+str(i)
# 	setattr(functions,fname,psi(i))
# 	#globals()[fname]=psi(i)

functions.square=lambda y:y**2




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


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

psis=[psi(i) for i in range(1,11)]

def genpsi(d,ijk):
	#psis=[getattr(examplefunctions,'psi{}'.format(i)) for i in range(1,11)]
	def psi_ijk(X):
		out=1
		for k,l in zip(ijk,range(d)):
			out*=psis[k](X[:,l])
		return out
	return psi_ijk

for d in [1,2,3]:
	for i,ijk in enumerate(gen_n_dtuples(10,d)):
		psi=genpsi(d,ijk)
		setattr(functions,'psi{}_{}d'.format(i+1,d),psi)







#----------------------------------------------------------------------------------------------------
# test
#----------------------------------------------------------------------------------------------------



def test():
	for p in H_coefficients_list: print(p)

	





