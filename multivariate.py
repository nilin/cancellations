# nilin




import jax.numpy as jnp
import jax
import util

import math
import jax.random as rnd

import config as cfg

import pdb


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




def gen_lossgrad(f,lossfn=cfg.lossfn):

	@jax.jit
	def collectiveloss(params,X,Y):
		fX=f(params,X)
		return lossfn(fX,Y)

	@jax.jit	
	def lossgrad(params,X,Y):
		return jax.value_and_grad(collectiveloss)(params,X,Y)

	return lossgrad
	

#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------



"""
# computes widths[-1] functions
"""
def initweights_NN(widths,key):


	k1,*Wkeys=rnd.split(key,100)
	k2,*bkeys=rnd.split(key,100)

	Ws=[rnd.normal(key,(m2,m1))/math.sqrt(m1) for m1,m2,key in zip(widths[:-1],widths[1:],Wkeys)]
	bs=[rnd.normal(key,(m,)) for m,key in zip(widths[1:-1],bkeys)]

	return [Ws,bs]
