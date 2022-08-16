import jax
import jax.numpy as jnp
import numpy as np
import multivariate as mv
import util
import AS_tools
import pdb


def staticSlater(F):
	return util.noparams(AS_tools.Slater(util.dummyparams(F)))

	
#----------------------------------------------------------------------------------------------------
# Hermite polynomials
#----------------------------------------------------------------------------------------------------

			
def hermitecoefficients(n,convention):
	return He_coefficients(n) if convention=='He' else H_coefficients(n)

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


def hermitecoefficientblock(n,convention):
	return jnp.array([p+[0]*(n+1-len(p)) for p in hermitecoefficients(n,convention)])





#----------------------------------------------------------------------------------------------------

def HermiteSlater(n,convention,envelopevariance):

	envelope_singlesample=lambda x:jnp.exp(-jnp.sum(jnp.square(x))/(2*envelopevariance))
	envelope=jax.vmap(envelope_singlesample)
	AF=staticSlater(genhermitefunctions(n-1,convention))

	@jax.jit
	def AF_(X):
		return envelope(X)*AF(X)

	return AF_


def genhermitefunctions(n,convention):
	coefficients=hermitecoefficientblock(n,convention)
	return mv.genpolynomialfunctions(coefficients)	






#----------------------------------------------------------------------------------------------------
# Gaussians
#----------------------------------------------------------------------------------------------------

def Gaussian(mean,var=None,std=None):

	if var==None:
		var=std**2

	@jax.jit
	def f(X):
		S=jnp.sum((X-mean[None,:])**2,axis=-1)
		return jnp.exp(-S/(2*var))

	return f

def GaussianSlater1D(n):
	std=1/n
	means=jnp.expand_dims(np.arange(-1,1.01,2/(n-1))/(1+3/(n-1)),axis=-1)
	functions=[Gaussian(mean,std=std) for mean in means]

	def F(X):
		Y=jnp.stack([f(X) for f in functions],axis=-1)
		return Y

	return staticSlater(F)



def test():

	import jax.random as rnd
	import config as cfg

	X=rnd.uniform(cfg.nextkey(),(1,5,1),minval=-1,maxval=1)
	f,fs=GaussianSlater1D(5)	

	print(X)
	for f in fs:
		print([f(X[:,i,:]) for i in range(5)])
	print(f(X))
