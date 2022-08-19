import jax
import jax.numpy as jnp
import numpy as np
import multivariate as mv
import util
import AS_tools
import pdb
import itertools



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
		
def polynomial_product(ps,X):
	return util.prod([polynomial(p,X[:,i]) for i,p in enumerate(ps)])

def get_Slater_poly_products(n):
	return AS_tools.gen_Slater(n,polynomial_product)

	
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
	

def hermiteSlater(n,d,envelopevariance):

	envelope=lambda x:jnp.exp(-jnp.sum(jnp.square(x),axis=(-2,-1))/(2*envelopevariance))
	p_nd=hermite_nd_params(n,d)
	Slater_poly_products=get_Slater_poly_products(n)

	@jax.jit
	def AF_(X):
		return envelope(X)*Slater_poly_products(p_nd,X)

	return AF_



#----------------------------------------------------------------------------------------------------
# Gaussians
#----------------------------------------------------------------------------------------------------

def isoGaussian(var=None,std=None):

	if var==None:
		var=std**2

	@jax.jit
	def f(mean,X):
		S=jnp.sum((X-mean[None,:])**2,axis=-1)
		return jnp.exp(-S/(2*var))

	return f

def packpoints(k):
	r=1/k
	centers=np.arange(-1+r,1,2*r)
	return centers,r

def gaussianSlater(n,d):

	tups=gen_n_dtuples(n,d)
	k=max([max(t) for t in tups])+1

	means1d,r=packpoints(k)
	means=[means1d[t] for t in tups]	

	return util.fixparams(AS_tools.gen_Slater(n,isoGaussian(std=r)),means)


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

#def test():
#
#	import jax.random as rnd
#	import config as cfg
#
#	X=rnd.uniform(cfg.nextkey(),(1,5,1),minval=-1,maxval=1)
#	f,fs=GaussianSlater1D(5)	
#
#	print(X)
#	for f in fs:
#		print([f(X[:,i,:]) for i in range(5)])
#	print(f(X))


if __name__=='__main__':

	print(sumsto(3,2))
	print()
	print(gen_n_dtuples(5,2))
	print()
	print(n_dtuples_maxdegree(5,2))
	print()
	for particle in Hermite_nd_params(5,2):
		print(particle)
	print()
	for p in H_coefficients_list[:5]: print(p)

	






"""
#def genericpolynomialfunctions(degree):		#coefficients dimensions: function,degree
#	monos=genmonomialfunctions(degree)
#
#	@jax.jit
#	def P(coefficients,x):
#		return jnp.inner(monos(x),coefficients)
#	return P
#
# def genmonomialfunctions(n):
# 	@jax.jit
# 	def F(x):
# 		x=jnp.squeeze(x)
# 		xk=jnp.ones_like(x)
# 		out=[]
# 		for k in range(n+1):
# 			out.append(xk)	
# 			xk=x*xk
# 		return jnp.stack(out,axis=-1)
# 	return F
# 
# def polynomial(coefficients,X):
# 	n=len(coefficients)-1
# 	jnp.inner(monomials(X,n),coefficients)
# 
# 
# def genpolynomialfunctions(coefficients):	#coefficients dimensions: function,degree
# 	degree=coefficients.shape[1]-1
# 	P_=genericpolynomialfunctions(degree)
# 
# 	@jax.jit
# 	def P(x):
# 		return P_(coefficients,x)
# 
# 	return P
#
#def hermitecoefficients(n,convention):
#	return He_coefficients(n) if convention=='He' else H_coefficients(n)
#
#def hermitecoefficientblock(n,convention):
#	return jnp.array([p+[0]*(n+1-len(p)) for p in hermitecoefficients(n,convention)])
#
#def genhermitefunctions(n,convention):
#	coefficients=hermitecoefficientblock(n,convention)
#	return mv.genpolynomialfunctions(coefficients)	
#
#def staticSlater(F):
#	return util.noparams(AS_tools.Slater(util.dummyparams(F)))
#
#
#def HermiteSlater(n,convention,envelopevariance):
#
#	envelope_singlesample=lambda x:jnp.exp(-jnp.sum(jnp.square(x))/(2*envelopevariance))
#	envelope=jax.vmap(envelope_singlesample)
#	AF=staticSlater(genhermitefunctions(n-1,convention))
#
#	@jax.jit
#	def AF_(X):
#		return envelope(X)*AF(X)
#
#	return AF_
"""
