# nilin




import jax.numpy as jnp
import jax
import util

import math
import jax.random as rnd

import config as cfg
from util import activations
import pdb


#=======================================================================================================
# NN 
#=======================================================================================================




def gen_NN_wideoutput(ac):
	activation=activations[ac]

	@jax.jit
	def NN(params,X):
		Ws,bs=params
		for W,b in zip(Ws[:-1],bs):
			X=jnp.inner(X,W)+b[None,:]
			X=activation(X)
		return jnp.inner(X,Ws[-1])

	return NN


def gen_NN(activation):
	return util.scalarfunction(gen_NN_wideoutput(activation))




def gen_skip_NN_NS(ac):
	activation=activations[ac]

	@jax.jit
	def NN(params,X):
		Ws,bs=params

		X=util.collapselast(X,2)
		X=jnp.inner(X,Ws[0])+bs[0][None,:]
		X=activation(X)
		for W,b in zip(Ws[1:-1],bs[1:]):
			skip=X
			X=jnp.inner(X,W)+b[None,:]
			X=activation(X)+X
		return jnp.squeeze(jnp.inner(X,Ws[-1]))

	return NN



def gen_NN_NS(activation):
	NN=gen_NN(activation)

	@jax.jit
	def NN_NS(params,X):
		X=util.collapselast(X,2)
		return NN(params,X)

	return NN_NS




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




def gen_lossgrad(f,lossfn=None):

	if lossfn==None: lossfn=cfg.getlossfn()


	@jax.jit
	def collectiveloss(params,X,*Y):
		fX=f(params,X)
		return lossfn(fX,*Y)

	@jax.jit	
	def lossgrad(params,X,*Y):
		return jax.value_and_grad(collectiveloss)(params,X,*Y)

	return lossgrad
	

#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------



"""
# computes widths[-1] functions
"""
def initweights_NN(widths):

	key=cfg.nextkey()

	k1,*Wkeys=rnd.split(key,100)
	k2,*bkeys=rnd.split(key,100)

	Ws=[rnd.normal(key,(m2,m1))/math.sqrt(m1) for m1,m2,key in zip(widths[:-1],widths[1:],Wkeys)]
	bs=[rnd.normal(key,(m,))*cfg.biasinitsize for m,key in zip(widths[1:-1],bkeys)]

	return [Ws,bs]



#----------------------------------------------------------------------------------------------------
# operations on functions
#----------------------------------------------------------------------------------------------------


def sum_f(f):
	
	@jax.jit
	def sf(paramsbundle,X):
		out=0
		for params in paramsbundle:
			out=out+f(params,X)
		return out

	return sf


def product_f(f):
	
	@jax.jit
	def pr(paramsbundle,X):
		out=0
		for i,params in enumerate(paramsbundle):
			out=out*f(params,X[:,i,:])
		return out

	return pr



def add_static_f(*fs):

	@jax.jit
	def sumf(X):
		out=0
		for f in fs:
			out=out+f(X)
		return out
	return sumf


		












