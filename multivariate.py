# nilin




import jax.numpy as jnp
import jax
import util

import math
import jax.random as rnd





#=======================================================================================================
# NN 
#=======================================================================================================


activation=util.ReLU

@jax.jit
def NN(params,X):
	Ws,bs=params
	for W,b in zip(Ws[:-1],bs):
		X=jnp.inner(X,W)+b[None,:]
		X=activation(X)
	return jnp.squeeze(jnp.inner(X,Ws[-1]))


@jax.jit
def NN_NS(params,X):
	X=util.collapselast(X,2)
	return NN(params,X)




#----------------------------------------------------------------------------------------------------
# polynomials
#----------------------------------------------------------------------------------------------------



def genmonomialfunctions(n):

	@jax.jit
	def F(x):
		x=jnp.squeeze(x)
		xk=jnp.ones_like(x)
		out=[]
		for k in range(n+1):
			out.append(xk)	
			xk=x*xk
		return jnp.stack(out,axis=-1)
	return F
		

def genericpolynomialfunctions(degree):		#coefficients dimensions: function,degree
	monos=genmonomialfunctions(degree)

	@jax.jit
	def P(coefficients,x):
		return jnp.inner(monos(x),coefficients)
	return P

def genpolynomialfunctions(coefficients):	#coefficients dimensions: function,degree
	degree=coefficients.shape[1]-1
	P_=genericpolynomialfunctions(degree)

	@jax.jit
	def P(x):
		return P_(coefficients,x)

	return P





	

#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------



def genW(k0,n,d,widths):

	if type(widths)!=list:
		print('Casting width to singleton list')
		widths=[widths]

	k1,*Wkeys=rnd.split(k0,100)
	k2,*bkeys=rnd.split(k0,100)

	Ws=[rnd.normal(key,(m2,m1))/math.sqrt(m1) for m1,m2,key in zip([n*d]+widths,widths+[1],Wkeys)]
	bs=[rnd.normal(key,(m,)) for m,key in zip(widths,bkeys)]

	return [Ws,bs]
