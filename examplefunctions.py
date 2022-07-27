import jax
import jax.numpy as jnp
import multivariate as mv
import util
import AS_tools



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

	@jax.jit
	def AF_(X):
		AF=staticSlater(genhermitefunctions(n-1,convention))
		return envelope(X)*AF(X)

	return AF_


def genhermitefunctions(n,convention):
	coefficients=hermitecoefficientblock(n,convention)
	return mv.genpolynomialfunctions(coefficients)	






