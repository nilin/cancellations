# nilin




import jax.numpy as jnp
import jax
from ..utilities import arrayutil, tracking,config as cfg

import math
import jax.random as rnd

from ..utilities.arrayutil import activations
from ..utilities import arrayutil
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
	return arrayutil.scalarfunction(gen_NN_wideoutput(activation))


def gen_NN_NS(activation):
	NN=gen_NN(activation)

	@jax.jit
	def NN_NS(params,X):
		X=arrayutil.collapselast(X,2)
		return NN(params,X)

	return NN_NS




def gen_lossgrad(f,lossfn=None):
	if lossfn==None: lossfn=cfg.getlossfn()

	def collectiveloss(params,X,*Y):
		return lossfn(f(params,X),*Y)

	return jax.value_and_grad(collectiveloss)
	

#----------------------------------------------------------------------------------------------------
# random initializations
#----------------------------------------------------------------------------------------------------


def initweights_NN(widths,*args,**kw):
	ds=widths
	Ws=[arrayutil.initweights((d2,d1)) for d1,d2 in zip(ds[:-1],ds[1:])]
	bs=[rnd.normal(tracking.nextkey(),(d2,))*cfg.biasinitsize for d2 in ds[1:]]

	return list(zip(Ws,bs))


#----------------------------------------------------------------------------------------------------
# operations on functions
#----------------------------------------------------------------------------------------------------

def multiply(*fs):
	if max([arrayutil.takesparams(f) for f in fs]):
		def F(paramsbundle,X):
			out=1
			for f,params in zip(fs,paramsbundle):
				out*=arrayutil.pad(f)(params,X)
			return out
	else:
		def F(X):
			out=1
			for f in fs: out*=f(X)
			return out
	return F




def inspect_composition(steps,params,X):
	layers=[(None,X)]
	for step,weights in zip(steps,params):
		try:
			layers.append((weights,step(weights,layers[-1][-1])))
		except Exception as e:
			layers.append((weights,None))
	return layers



