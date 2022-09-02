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




def gen_NN_NS(activation):
	NN=gen_NN(activation)

	@jax.jit
	def NN_NS(params,X):
		X=util.collapselast(X,2)
		return NN(params,X)

	return NN_NS





#----------------------------------------------------------------------------------------------------




def gen_lossgrad(f,lossfn=None):
	if lossfn==None: lossfn=cfg.getlossfn()

	def collectiveloss(params,X,*Y):
		return lossfn(f(params,X),*Y)

	return jax.value_and_grad(collectiveloss)
#	l_grad=jax.value_and_grad(collectiveloss)
#
#	@jax.jit	
#	def lossgrad(params,X,*Y):
#		return l_grad(params,X,*Y)
#
#	return lossgrad


	

#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------


def initweights_NN(widths,*args,**kw):
	ds=widths
	Ws=[util.initweights((d2,d1)) for d1,d2 in zip(ds[:-1],ds[1:])]
	bs=[rnd.normal(cfg.nextkey(),(d2,))*cfg.biasinitsize for d2 in ds[1:]]

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
	



