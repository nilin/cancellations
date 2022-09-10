import jax
import jax.numpy as jnp
import numpy as np
from ..functions import multivariate as mv
import util
import AS_tools
import pdb
import itertools
import functions



#----------------------------------------------------------------------------------------------------
# polynomials
#----------------------------------------------------------------------------------------------------

def monomials(x,n):
	x=jnp.squeeze(x)
	xk=jnp.ones_like(x)
	out=[]
	for k in range(n+1):
		out.append(xk)	
		xk=x*xk
	return jnp.stack(out,axis=-1)

def polynomial(coefficients,X):
	n=len(coefficients)-1
	return jnp.inner(monomials(X,n),coefficients)
		
#def polynomial_product(pi,X):
#	return util.prod([polynomial(pij,X[:,j]) for j,pij in enumerate(pi)])

def polynomial_products(ps,X):
	return jnp.stack([util.prod([polynomial(pij,X[:,j]) for j,pij in enumerate(pi)]) for pi in ps],axis=-1)
	
#----------------------------------------------------------------------------------------------------
# Hermite polynomials
#----------------------------------------------------------------------------------------------------

			
def He_coefficients(n):
	if n==0:
		return [[1]]
	if n==1:
		return [[1],[0,1]]
	else:
		A=He_coefficients(n-1)
		a1,a2=A[-1],A[-2]+2*[0]
		a=[-(n-1)*a2[0]]
		for k in range(1,n+1):
			a.append(a1[k-1]-(n-1)*a2[k])
		A.append(a)
		return A

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


He_coefficients_list=[jnp.array(p) for p in He_coefficients(25)]
H_coefficients_list=[jnp.array(p) for p in H_coefficients(25)]



def hermite_nd_params(n,d):		
	return [[H_coefficients_list[p] for p in phi] for phi in gen_n_dtuples(n,d)]	

def gen_hermitegaussproducts(n,d,envelopevariance=1):
	hermiteprods=util.fixparams(polynomial_products,hermite_nd_params(n,d))	
	envelope=lambda x:jnp.exp(-jnp.sum(jnp.square(x),axis=(-1))/(2*envelopevariance))
	return jax.jit(lambda X:envelope(X)[:,None]*hermiteprods(X))

#def gen_hermitegaussproducts_separate(n,d,envelopevariance=1):
#	hermiteprods=[util.fixparams(polynomial_product,pi) for pi in hermite_nd_params(n,d)]
#	envelope=lambda x:jnp.exp(-jnp.sum(jnp.square(x),axis=(-1))/(2*envelopevariance))
#	return [jax.jit(lambda X:envelope(X)*prod(X)) for prod in hermiteprods]

#----------------------------------------------------------------------------------------------------
# Gaussians
#----------------------------------------------------------------------------------------------------

def isoGaussian(var=None,std=None):

	if var==None:
		var=std**2

	@jax.jit
	def f(mean,X):
		S=jnp.sum((X[:,None,:]-mean[None,:,:])**2,axis=-1)
		return jnp.exp(-S/(2*var))

	return f

def packpoints(k):
	r=1/k
	centers=np.arange(-1+r,1,2*r)
	return centers,r

def gen_parallelgaussians(n,d):
	tups=gen_n_dtuples(n,d)
	k=max([max(t) for t in tups])+1

	means1d,r=packpoints(k)
	means=jnp.array([means1d[t] for t in tups])
	return util.fixparams(isoGaussian(std=r),means)


#----------------------------------------------------------------------------------------------------
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
# test
#----------------------------------------------------------------------------------------------------






#def hermiteSlater(n,d,std=1):
#	return functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen')
#
#def gaussianSlater(n,d):
#	return functions.Slater('parallelgaussians',n=n,d=d,mode='gen')









if __name__=='__main__':

#	print(sumsto(3,2))
#	print()
#	print(gen_n_dtuples(5,2))
#	print()
#	print(n_dtuples_maxdegree(5,2))
#	print()
#	for particle in Hermite_nd_params(5,2):
#		print(particle)
#	print()
	for p in H_coefficients_list[:5]: print(p)

	





