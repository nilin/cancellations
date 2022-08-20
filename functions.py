# nilin

import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
#from GPU_sum import sum_perms_multilayer as sumperms
import optax
import math
import testing
import AS_tools
import AS_tools as ASt
import AS_HEAVY
import multivariate as mv
import jax.random as rnd
import examplefunctions
import learning as lrn
import pdb
import backflow as bf
import copy
from inspect import signature
import importlib






class ParameterizedFunc:

	def __init__(self,**kw):
		self.kw=kw
		for k,v in kw.items():
			setattr(self,k,v)
		self.restore()

	def get_lossgrad(self,lossfn=None):
		return mv.gen_lossgrad(self.f,lossfn=lossfn)

	def fwithparams(self,params):
		fs=util.fixparams(self.f,params)
		return util.makeblockwise(fs)

	# for pickling
	def compress(self):
		self.f=None
		return self

	def restore(self):
		if 'f' not in vars(self) or self.f==None:
			self.f=self._gen_f_()
		return self

	def typename(self):
		return type(self).__name__

	def info(self):
		return self.typename()+'\n'+'\n'.join([str(k,'=',v) for k,v in self.kw.items()])

	def getclone(self):
		return copy.deepcopy(self)

	def _gen_f_(self):
		return mv.pad(self.gen_f())

	# def get_f(self):


class FunctionDescription(ParameterizedFunc):

	def __init__(self,**kw):
		super().__init__(**kw)
		self.weights=self.initweights()

	def eval(self,X):
		return self.fwithparams(self.weights)(X)
	
	def initweights(self):
		return globals()['initweights_{}'.format(type(self).__name__)](**self.kw)

	def rinse(self):
		self.weights=None
		return self

	def getemptyclone(self):
		return self.getclone().rinse()


class ComposedFunction(FunctionDescription):
	def __init__(self,*elements):
		super().__init__(elements=[cast(e) for e in elements])

	def gen_f(self):
		return util.compose(*[e._gen_f_() for e in self.elements])

	def initweights(self):
		return [e.initweights() for e in self.elements]

	def typename(self):
		return '-'.join([e.typename() for e in self.elements])

	def info(self):
		return '\n\n'.join([cfg.indent(e.info()) for e in self.elements])



class NNfunction(FunctionDescription):
	def typename(self):
		return self.activation+type(self).__name__




class ASNN(NNfunction):
	def gen_f(self):
		NN_NS=mv.gen_NN_NS(self.activation)
		return ASt.gen_Af(self.n,NN_NS)
	
class Backflow0(NNfunction):
	def gen_f(self):	
		return ASt.gen_backflow0(activation=self.activation)

class Backflow1(NNfunction):
	def gen_f(self):	
		return ASt.gen_backflow1(activation=self.activation)

class SingleparticleNN(NNfunction):
	def gen_f(self):
		return bf.gen_singleparticleNN(activation=self.activation)



#=======================================================================================================

class Slater(FunctionDescription):
	def __init__(self,basisfunctions,**kw):
		super().__init__(basisfunctions=cast(basisfunctions,**kw))

	def gen_f(self):
		parallel=jax.vmap(self.basisfunctions._gen_f_(),in_axes=(None,-2),out_axes=-2)
		return jax.jit(lambda params,X: jnp.linalg.det(parallel(params,X)))

	def initweights(self):
		return self.basisfunctions.initweights()
		

class Wrappedfunction(FunctionDescription):
	def __init__(self,gf,**kw):
		self.globalgen_f=gf
		super().__init__(**kw)

	def gen_f(self):
		return globals()[self.globalgen_f](**self.kw)

	def initweights(self):
		return None



from examplefunctions import hermitegaussproducts,parallelgaussians

def hermiteSlater(n,d,std=1/8):
	return Slater('hermitegaussproducts',n=n,d=d)

def gaussianSlater(n,d):
	return Slater('parallelgaussians',n=n,d=d)





#=======================================================================================================
# init NN
#=======================================================================================================

def initweights_ASNN(n,d,widths,**kw):
	widths[0]=n*d
	return mv.initweights_NN(widths)



def initweights_Backflow0(n,d,widths,**kw):
	ds,k=widths
	ds[0]=d
	return [bf.initweights_backflow(ds),util.initweights((k,n,ds[-1]))]

def initweights_Backflow1(n,d,widths,**kw):
	ds,k=widths
	ds[0]=d
	return [bf.initweights_backflow(ds),[util.initweights((k,n,ds[-1])),util.initweights((ds[-1],))]]
"""
#==================
# example: ferminet
#==================
#
#def initweights_ferminet(widths,**kw):
#	ds0,ds1,k=widths
#	return bf.initweights_FN_backflow([ds0,ds1])+[util.initweights((k,))]
#
"""

def initweights_SingleparticleNN(d,widths,**kw):
	widths[0]=d
	return mv.initweights_NN(widths)



#=======================================================================================================



def cast(f,**kw):
	return f if isinstance(f,ParameterizedFunc) else Wrappedfunction(f,**kw)

def getfunc(ftype,**kw):
	return globals()[ftype](**kw)

