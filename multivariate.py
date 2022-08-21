# nilin




import jax.numpy as jnp
import jax
import util

import math
import jax.random as rnd

import config as cfg
from util import activations
import pdb

from inspect import signature

#=======================================================================================================
# NN 
#=======================================================================================================

def gen_NN_layer(ac):
	activation=activations[ac]

	@jax.jit
	def f(Wb,X):
		W,b=Wb[0],Wb[1]
		return activation(jnp.inner(X,W)+b[None,:])
	return f
		

def gen_NN_wideoutput(ac):
	L=gen_NN_layer(ac)

	@jax.jit
	def NN(params,X):
		for Wb in params:
			X=L(Wb,X)
		return X

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




def gen_lossgrad_batchfirst(f,lossfn):

	def collectiveloss(params,X,*Y):
		return lossfn(f(params,X),*Y)

	l_grad=jax.value_and_grad(collectiveloss)

	@jax.jit	
	def lossgrad(params,X,*Y):
		return l_grad(params,X,*Y)

	return lossgrad

#
#def gen_lossgrad_batchlast(f,lossfn):
#
#	def singlesampleloss(params,x,*ys):
#		X=jnp.expand_dims(x,axis=0)
#		Ys=[jnp.expand_dims(y,axis=0) for y in ys]
#		return lossfn(f(params,X),*Ys)
#
#	singlesample_l_grad=jax.value_and_grad(singlesampleloss)
#	#parallel_l_grad=[jax.vmap(singlesample_l_grad,in_axes=(None,0)+ntargets*(0,),out_axes=(0,0)) for ntargets in range(2)]
#	parallel_l_grad=[jax.vmap(singlesample_l_grad,in_axes=(None,0)+ntargets*(0,)) for ntargets in range(2)]
#
#	@jax.jit	
#	def lossgrad(params,X,*Y):
#		losses,grads=parallel_l_grad[len(Y)](params,X,*Y)
#		loss,grad=jnp.average(losses),util.applyonleaves(grads,lambda A:jnp.average(A,axis=0))
#		#util.printshape(loss)
#		#util.printshape(grad)
#		return loss,grad
#
#	return lossgrad
#


def gen_lossgrad(f,lossfn=None,batchmode='first'):
	if lossfn==None: lossfn=cfg.getlossfn()

	if batchmode=='first':
		return gen_lossgrad_batchfirst(f,lossfn)
	if batchmode=='last':
		return gen_lossgrad_batchlast(f,lossfn)
	
		
	

#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------


"""
# computes widths[-1] functions
"""
def initweights_NN(widths,*args,**kw):

	key=cfg.nextkey()

	k1,*Wkeys=rnd.split(key,100)
	k2,*bkeys=rnd.split(key,100)

	Ws=[rnd.normal(key,(m2,m1))/math.sqrt(m1) for m1,m2,key in zip(widths[:-1],widths[1:],Wkeys)]
	bs=[rnd.normal(key,(m,))*cfg.biasinitsize for m,key in zip(widths[1:],bkeys)]

	return list(zip(Ws,bs))




#----------------------------------------------------------------------------------------------------
# operations on functions
#----------------------------------------------------------------------------------------------------

def multiply(*fs):
	if max([takesparams(f) for f in fs]):
		def F(paramsbundle,X):
			out=1
			for f,params in zip(fs,paramsbundle):
				out*=pad(f)(params,X)
			return out
	else:
		def F(X):
			out=1
			for f in fs: out*=f(X)
			return out
	return F


def takesparams(f):
	return len(signature(f).parameters)==2	

def pad(f):
	return f if takesparams(f) else util.dummyparams(f)
	



