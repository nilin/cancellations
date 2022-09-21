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
	p=mathutil.fixparams(polynomial,H_coefficients_list[n])
	psi_n=jax.jit(lambda x: jnp.exp(-x**2/2)*p(x))
	return psi_n

def totalenergy(n): return sum([i+1/2 for i in range(n)])



# load function definitions

for i in range(10):
	setattr(functions,'psi'+str(i),psi(i))
	globals()['psi'+str(i)]=psi(i)

functions.square=lambda y:y**2

#----------------------------------------------------------------------------------------------------
# test
#----------------------------------------------------------------------------------------------------



def test():
	for p in H_coefficients_list: print(p)

	





