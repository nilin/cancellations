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
from functions import multivariate as mv
import jax.random as rnd
import backflow as bf
import copy
from collections import deque
import backflow
from display import dash
from backflow import gen_backflow,initweights_Backflow
from AS_tools import detsum #,initweights_detsum
from jax.numpy import tanh
from util import drelu
import examplefunctions
from examplefunctions import gen_parallelgaussians,gen_hermitegaussproducts


class FunctionDescription:

	def __init__(self,initweights=True,**kw):
		self.kw=kw
		for k,v in kw.items():
			setattr(self,k,v)
		self.restore()
		if initweights: self.initweights()

	def get_lossgrad(self,lossfn=None):
		return mv.gen_lossgrad(self.f,lossfn=lossfn)

	def fwithparams(self,params):
		return util.fixparams(self.f,params)

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
			return util.pad(self.gen_f())

	# parameters

	def initweights(self):
		self.weights=self._initweights_(**self.kw)

	def eval(self,X,blocksize=10**5,**kw):
		return util.eval_blockwise(self.f,self.weights,X,blocksize=blocksize,**kw)

	@staticmethod	
	def _initweights_(**kw):
		return None

	def rinse(self):
		self.weights=None
		return self

	def popweights(self):
		weights=self.weights
		self.weights=None
		return weights

	def getemptyclone(self):
		c=copy.deepcopy(self)
		return c.rinse()

	def compose(self,fd):
		return ComposedFunction(self,fd)

	# debug

	def _inspect_(self,params,X):
		return self.typename(),['input',params],[X,self._gen_f_()(params,X)]
#
#	def inspect(self,*args):
#		match len(args):
#			case 1: params,X=(self.weights,)+args
#			case 2: params,X=args
#		return (self.typename(),self._inspect_(params,X))



def initweights(*functions):
	for f in functions: f.initweights()


class ComposedFunction(FunctionDescription):
	def __init__(self,*nestedelements):
		elements=[e for E in nestedelements for e in (E.elements if isinstance(E,ComposedFunction) else [E])]
		weights=[w for E in nestedelements for w in (E.popweights() if isinstance(E,ComposedFunction) else [cast(E).popweights()])]
		super().__init__(elements=[cast(e).compress() for e in elements],initweights=False)
		self.weights=weights

	def gen_f(self):
		return util.compose(*[e._gen_f_() for e in self.elements])

#	def inheritweights(self):
#		self.weights=[e.popweights() for e in self.elements]

	@staticmethod
	def _initweights_(elements):
		return [e._initweights_(**e.kw) for e in elements]

	def typename(self):
		return ' -> '.join([e.richtypename() for e in self.elements])+' composition'

	def info(self):
		return '\n'+'\n\n'.join([cfg.indent(e.getinfo()) for e in self.elements])

	def compress(self):
		c=super().compress()
		new_c_elements=[e.compress() for e in c.elements]
		c.elements=new_c_elements
		return c

	def _inspect_(self,params,X):
		subprocs=['input']
		Xhist=[X]
		for e,w in zip(self.elements,params):
			fname,subproc,Xsubproc=e._inspect_(w,Xhist[-1])
			subprocs.append((fname,subproc,Xsubproc))
			Xhist.append(Xsubproc[-1])
			#try:
			#	subprocs.append(e._inspect_(w,Xhist[-1]))
			#	Xhist.append(subprocs[-1][-1])
			#except Exception as e:
			#	cfg.dblog(str(e))
			#	break
		return self.typename(),subprocs,Xhist


#=======================================================================================================
class NNfunction(FunctionDescription):
	def richtypename(self):
		return self.activation+'-'+self.typename()

	@staticmethod
	def _initweights_(widths,**kw):
		return mv.initweights_NN(widths)

#=======================================================================================================

class Equivariant(FunctionDescription):
	pass

class SingleparticleNN(NNfunction,Equivariant):
	def gen_f(self):
		return bf.gen_singleparticleNN(activation=self.activation)

	@staticmethod
	def _initweights_(widths,**kw):
		return mv.initweights_NN(widths)


class Backflow(NNfunction,Equivariant):
	def gen_f(self):	
		return bf.gen_backflow(self.activation)

	@staticmethod
	def _initweights_(widths,**kw):
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
	def switch(self,newclass):
		kw={newclass.translation(k):v for k,v in self.kw.items()}
		Tf=newclass(**kw)
		if hasattr(self,'weights') and self.weights!=None: Tf.weights=self.weights
		return Tf
	@staticmethod
	def translation(k): return k

class Nonsym(FunctionDescription,Switchable):
	def antisym():
		return globals()[self.antisymtype]
	def getantisym(self):
		return self.switch(self.antisym())
	switchtype=getantisym

class NN(NNfunction,Nonsym):
	antisymtype='ASNN'
	def gen_f(self): return mv.gen_NN_NS(self.activation)

	@staticmethod
	def _initweights_(n,d,widths,**kw):
		widths[0]=n*d
		return mv.initweights_NN(widths)

class ProdSum(Nonsym):
	antisymtype='DetSum'
	def gen_f(self): return AS_tools.prodsum
	@staticmethod
	def _initweights_(k,n,d,**kw):
		return util.initweights((k,n,d))
	@staticmethod
	def translation(name):return 'k' if name=='ndets' else name

class ProdState(Nonsym):
	antisymtype='Slater'
	def __init__(self,basisfunctions,**kw):
		super().__init__(basisfunctions=cast(basisfunctions,**kw).compress())

	def gen_f(self):
		parallel=jax.vmap(self.basisfunctions._gen_f_(),in_axes=(None,-2),out_axes=-2)
		return jax.jit(lambda params,X: ASt.diagprods(parallel(params,X)))

	@staticmethod
	def _initweights_(basisfunctions):
		return basisfunctions._initweights_(**basisfunctions.kw)

	def richtypename(self): return self.basisfunctions.richtypename()+dash+self.typename()
	def info(self): return cfg.indent(self.basisfunctions.info())

#=======================================================================================================

class Antisymmetric(FunctionDescription,Switchable):
	def getnonsym(self):
		return self.switch(self.nonsym)
	switchtype=getnonsym

	@classmethod
	def _initweights_(cls,**kw):
		return cls.nonsym._initweights_(**{cls.nonsym.translation(k):v for k,v in kw.items()})

class ASNN(Antisymmetric,NNfunction):
	nonsym=NN
	def gen_f(self):
		NN_NS=mv.gen_NN_NS(self.activation)
		return ASt.gen_Af(self.n,NN_NS)

class DetSum(Antisymmetric):
	nonsym=ProdSum
	def gen_f(self): return AS_tools.detsum
	@staticmethod
	def translation(name):return 'ndets' if name=='k' else name

	#def _inspect_(self,params,X): return AS_tools.inspectdetsum(params,X)

class Slater(Antisymmetric):
	nonsym=ProdState
	def __init__(self,basisfunctions,**kw):
		super().__init__(basisfunctions=cast(basisfunctions,**kw).compress())

	def gen_f(self):
		parallel=jax.vmap(self.basisfunctions._gen_f_(),in_axes=(None,-2),out_axes=-2)
		return jax.jit(lambda params,X: jnp.linalg.det(parallel(params,X)))

	def richtypename(self): return self.basisfunctions.richtypename()+dash+self.typename()
	def info(self): return cfg.indent(self.basisfunctions.info())


#=======================================================================================================

class Scalarfunction(FunctionDescription):
	pass

class Oddfunction(FunctionDescription):
	pass

class OddNN(NNfunction,Scalarfunction,Oddfunction):
	def gen_f(self):
		NN=mv.gen_NN(self.activation)
		scalarNN=lambda params,X:NN(params,jnp.expand_dims(X,axis=-1))
		return jax.jit(lambda params,X:scalarNN(params,X)-scalarNN(params,-X))

class Outputscaling(Scalarfunction,Oddfunction):
	def gen_f(self):
		return jax.jit(lambda c,X:c*X)
	@staticmethod
	def _initweights_():
		return 1.0

class Flatten(Scalarfunction,Oddfunction):
	def gen_f(self):
		return jax.jit(lambda _,Y:jnp.tanh(self.sharpness*Y))

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
	if type(f)==ComposedFunction:
		elements=[switchtype(e) if isinstance(e,Switchable) else e for e in f.elements]
		return ComposedFunction(*elements)
	elif isinstance(f,Switchable):
		return f.switchtype()
	else: raise ValueError




#=======================================================================================================



flatten=jax.jit(lambda y:jnp.tanh(10*y))


def cast(f,**kw):
	if type(f)==tuple:
		f0,kw0=f
		return cast(f0,**kw0)
	return f if isinstance(f,FunctionDescription) else Wrappedfunction(f,**kw)



def formatinspection(L):
	info,steps=L

	try:
		s='W.shape {}\n'.format(util.shapestr(info))
	except:
		s=info+'\n' if type(info)==str else ''

	try:
		v='Y.shape {}\n|'.format(str(steps.shape))
	except:
		if type(steps)==str:
			v=steps
		elif type(steps)==list:
			v=cfg.indent('\n'.join([formatinspection(step) for step in steps]))
		else:
			v='formatting failed'

	return s+v



def inspect(fd,X,formatarrays=None,msg=''):

	cfg.logcurrenttask('inspect function '+msg)

	if formatarrays==None:
		def formatarrays(name,val):
			s=name+' '
			if name in ['weights','X']:
				try: s+=util.shapestr(val)
				except Exception as e: s+='formatting error '+str(e)
			return s

	def steps(info,level=0):
		if type(info)==tuple and len(info)==3:
			fname,subprocs,Xhist=info
			S=[{'name':fname,'val':None,'level':level}]
			for subproc,X in zip(subprocs,Xhist):
				S+=steps(subproc,level=level+1)
				S.append({'name':'X','val':X,'level':level+1}) 
			return S

		else:
			if type(info)==str: return [{'name':info,'val':None,'level':level}]
			return [{'name':'weights','val':info,'level':level}] 


	S=steps(fd._inspect_(fd.weights,X))
	cfg.dblog(msg+'\n'+'\n'.join([step['level']*'| '+formatarrays(step['name'],step['val']) for step in S]))

	cfg.clearcurrenttask()
	return S
	

		