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
from dashboard import dash
import backflow
from backflow import gen_backflow,initweights_Backflow
from AS_tools import detsum #,initweights_detsum
from jax.numpy import tanh
from util import drelu
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

	def restore(self):
		self.f=self._gen_f_()
		return self

	def typename(self):
		return type(self).__name__

	def richtypename(self):
		return self.typename()

	def info(self):
		return ', '.join(['{}={}'.format(k,v) for k,v in self.kw.items()])

	def getinfo(self):
		return '{}\n{}'.format(self.richtypename(),self.info())

	def compress(self):
		c=copy.deepcopy(self)
		c.f=None
		return c

	def _gen_f_(self):
		if 'f' in vars(self) and self.f!=None:
			return self.f
		else:
			return mv.pad(self.gen_f())




class FunctionDescription(ParameterizedFunc):

	def __init__(self,initweights=True,**kw):
		super().__init__(**kw)
		self.weights=self.initweights(**kw) if initweights else None

	def eval(self,X):
		return self.fwithparams(self.weights)(X)

	@staticmethod	
	def initweights(**kw):
		return None

	def rinse(self):
		self.weights=None
		return self

	def getemptyclone(self):
		c=copy.deepcopy(self)
		return c.rinse()


class ComposedFunction(FunctionDescription):
	def __init__(self,*elements):
		elements=[e for E in elements for e in (E.elements if isinstance(E,ComposedFunction) else [E])]
		super().__init__(elements=[cast(e).compress() for e in elements])

	def gen_f(self):
		return util.compose(*[e._gen_f_() for e in self.elements])

	@staticmethod
	def initweights(elements):
		return [e.initweights(**e.kw) for e in elements]

	def typename(self):
		return ' -> '.join([e.richtypename() for e in self.elements])+' composition'

	def info(self):
		return '\n'+'\n\n'.join([cfg.indent(e.getinfo()) for e in self.elements])

	def compress(self):
		c=super().compress()
		new_c_elements=[e.compress() for e in c.elements]
		c.elements=new_c_elements
		return c


class NNfunction(FunctionDescription):
	def richtypename(self):
		return self.activation+'-'+self.typename()

#=======================================================================================================

class Equivariant(FunctionDescription):
	pass

class SingleparticleNN(NNfunction,Equivariant):
	def gen_f(self):
		return bf.gen_singleparticleNN(activation=self.activation)

	@staticmethod
	def initweights(widths,**kw):
		return mv.initweights_NN(widths)


class Backflow(NNfunction,Equivariant):
	def gen_f(self):	
		return bf.gen_backflow(self.activation)

	@staticmethod
	def initweights(widths,**kw):
		return bf.initweights_Backflow(widths)


import transformer
from transformer import initweights_SimpleSAB
from jax.nn import softmax

class SimpleSAB(Equivariant):
	def gen_f(self):
		d=self.d
		omega=lambda X:softmax(X/jnp.sqrt(d),axis=-1)
		return transformer.gen_simple_SAB(omega=omega)


#=======================================================================================================

class Switchable:
	def switch(self,mode,newclass):
		kw={self.translation(k):v for k,v in self.kw.items()}
		Tf=newclass(initweights=False,**self.kw)
		if mode=='tied': Tf.weights=self.weights
		elif mode=='copy': Tf.weights=copy.deepcopy(self.weights)
		elif mode=='empty': Tf.weights=None
		else: raise ValueError
		return Tf
	@staticmethod
	def translation(k): return k

class Nonsym(FunctionDescription,Switchable):
	def antisym():
		return globals()[self.antisymtype]
	def getantisym(self,mode):
		return self.switch(mode,self.antisym())
	switchtype=getantisym

class NN(NNfunction,Nonsym):
	antisymtype='ASNN'
	def gen_f(self): return mv.gen_NN_NS(self.activation)

	@staticmethod
	def initweights(n,d,widths,**kw):
		widths[0]=n*d
		return mv.initweights_NN(widths)

class ProdSum(Nonsym):
	antisymtype='DetSum'
	def gen_f(self): return AS_tools.prodsum
	@staticmethod
	def initweights(k,n,d,**kw): return util.initweights((k,n,d))
	@staticmethod
	def translation(name):return 'k' if name=='ndets' else name

class ProdState(Nonsym):
	antisymtype='Slater'
	def gen_f(self):
		parallel=jax.vmap(self.basisfunctions._gen_f_(),in_axes=(None,-2),out_axes=-2)
		return jax.jit(lambda params,X: ASt.diagprods(parallel(params,X)))

	@staticmethod
	def initweights(**kw):
		return self.basisfunctions.initweights(**kw)

	def richtypename(self): return self.basisfunctions.richtypename()+dash+self.typename()
	def info(self): return cfg.indent(self.basisfunctions.info())

#=======================================================================================================

class Antisymmetric(FunctionDescription,Switchable):
	def getnonsym(self,mode):
		return self.switch(mode,self.nonsym)
	switchtype=getnonsym

	@classmethod
	def initweights(cls,**kw):
		return cls.nonsym.initweights(**{cls.nonsym.translation(k):v for k,v in kw.items()})

class ASNN(NNfunction,Antisymmetric):
	nonsym=NN
	def gen_f(self):
		NN_NS=mv.gen_NN_NS(self.activation)
		return ASt.gen_Af(self.n,NN_NS)

class DetSum(Antisymmetric):
	nonsym=ProdSum
	def gen_f(self): return AS_tools.detsum
	@staticmethod
	def translation(name):return 'ndets' if name=='k' else name

class Slater(Antisymmetric):
	nonsym=ProdState
	def __init__(self,basisfunctions,**kw):
		super().__init__(basisfunctions=cast(basisfunctions,**kw).compress())

	def gen_f(self):
		parallel=jax.vmap(self.basisfunctions._gen_f_(),in_axes=(None,-2),out_axes=-2)
		return jax.jit(lambda params,X: jnp.linalg.det(parallel(params,X)))

	def richtypename(self): return self.basisfunctions.richtypename()+dash+self.typename()
	def info(self): return cfg.indent(self.basisfunctions.info())



#class BackflowAS(ComposedFunction,NNfunction):
#	def __init__(self,n,widths,k,activation,**kw):
#		super().__init__(\
#			Backflow(activation=activation,widths=widths,**kw),\
#			Wrappedfunction('detsum',n=n,ndets=widths[-1],k=k))

#=======================================================================================================




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

def switchtype(f:FunctionDescription):
	if isinstance(f,ComposedFunction):
		g=copy.deepcopy(f)
		newelements=[switchtype(e) if isinstance(e,Switchable) else e for e in g.elements]
		g.elements=newelements
		return g
	elif isinstance(f,Switchable):
		return f.switchtype('copy')
	else: raise ValueError


#def antisymmetrize(f:FunctionDescription):
#	if isinstance(f,ComposedFunction):
#		Af=copy.deepcopy(f)
#		newelements=[antisymmetrize(e) if isinstance(e,Nonsym) else e for e in Af.elements]
#		Af.elements=newelements
#		return Af
#	elif isinstance(f,Nonsym):
#		return f.getantisym('copy')
#	else: raise ValueError
#		
#def nonsymmetrize(Af:FunctionDescription):
#	if isinstance(Af,ComposedFunction):
#		f=copy.deepcopy(Af)
#		newelements=[nonsymmetrize(e) if isinstance(e,Antisymmetric) else e for e in f.elements]
#		f.elements=newelements
#		return f
#	elif isinstance(Af,Antisymmetric):
#		return Af.getnonsym('copy')
#	else: raise ValueError


#=======================================================================================================



flatten=jax.jit(lambda y:jnp.tanh(10*y))


def cast(f,**kw):
	if type(f)==tuple:
		f0,kw0=f
		return cast(f0,**kw0)
	return f if isinstance(f,ParameterizedFunc) else Wrappedfunction(f,**kw)


