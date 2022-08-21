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
import learning as lrn
import pdb
import backflow as bf
import copy
from inspect import signature
import importlib
from collections import deque

from backflow import gen_backflow,initweights_Backflow
from AS_tools import detsum,initweights_detsum
import examplefunctions
from examplefunctions import gen_parallelgaussians,gen_hermitegaussproducts


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
		return ', '.join(['{}={}'.format(k,v) for k,v in self.kw.items()])

	def getinfo(self):
		return '{}\n{}'.format(self.typename(),self.info())

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
		initname='initweights_{}'.format(del_ac(self.typename()))
		print(initname)
		if initname in globals():
			return globals()[initname](**self.kw)
		else:
			return None

	def rinse(self):
		self.weights=None
		return self

	def getemptyclone(self):
		return self.getclone().rinse()


class ComposedFunction(FunctionDescription):
	def __init__(self,*elements):
		elements=[e for E in elements for e in (E.elements if isinstance(E,ComposedFunction) else [E])]
		super().__init__(elements=[cast(e) for e in elements])

	def gen_f(self):
		return util.compose(*[e._gen_f_() for e in self.elements])

	def initweights(self):
		return [e.initweights() for e in self.elements]

	def typename(self):
		return '-'.join([e.typename() for e in self.elements])+' composition'

	def info(self):
		return '\n'+'\n\n'.join([cfg.indent(e.getinfo()) for e in self.elements])




#=======================================================================================================

class NNfunction(FunctionDescription):
	def typename(self):
		return self.activation+type(self).__name__


class ASNN(NNfunction):
	def gen_f(self):
		NN_NS=mv.gen_NN_NS(self.activation)
		return ASt.gen_Af(self.n,NN_NS)

class SingleparticleNN(NNfunction):
	def gen_f(self):
		return bf.gen_singleparticleNN(activation=self.activation)

class Backflow(NNfunction):
	def gen_f(self):	
		return bf.gen_backflow(self.activation)

class BackflowAS(ComposedFunction,NNfunction):
	def __init__(self,n,widths,k,activation,**kw):
		super().__init__(Backflow(activation=activation,widths=widths,**kw),Wrappedfunction('detsum',n=n,outdim=widths[-1],k=k))

class Slater(FunctionDescription):
	def __init__(self,basisfunctions,**kw):
		super().__init__(basisfunctions=cast(basisfunctions,**kw))

	def gen_f(self):
		parallel=jax.vmap(self.basisfunctions._gen_f_(),in_axes=(None,-2),out_axes=-2)
		return jax.jit(lambda params,X: jnp.linalg.det(parallel(params,X)))

	def initweights(self):
		return self.basisfunctions.initweights()

	def info(self):
		return cfg.indent(self.basisfunctions.getinfo())
		

class Wrappedfunction(FunctionDescription):
	def __init__(self,fname,mode=None,**kw):
		self.fname=fname
		self.mode=mode
		super().__init__(**kw)

	def gen_f(self):
		if self.mode=='gen':
			return globals()['gen_'+self.fname](**self.kw)
		else: return globals()[self.fname]

	def typename(self):
		return self.fname





#=======================================================================================================
# init NN
#=======================================================================================================

def initweights_ASNN(n,d,widths,**kw):
	widths[0]=n*d
	return mv.initweights_NN(widths)

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

#def initweights_SingleparticleNN(widths,**kw):
#	return mv.initweights_NN(widths)

initweights_SingleparticleNN=mv.initweights_NN

#=======================================================================================================



def cast(f,**kw):
	if type(f)==tuple:
		f0,kw0=f
		return cast(f0,**kw0)
	return f if isinstance(f,ParameterizedFunc) else Wrappedfunction(f,**kw)

def getfunc(ftype,**kw):
	return globals()[ftype](**kw)

def del_ac(s):
	for a in util.substringslast(util.activations):
		s=s.replace(a,'')
	return s
