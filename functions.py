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
from inspect import signature







class ParameterizedFunc:

	def __init__(self,fd):
		self.fdescr=fd
		self.restore()

	def getdescription(self):
		return self.fdescr

	def get_lossgrad(self,lossfn=None):
		return mv.gen_lossgrad(self.f,lossfn=lossfn)

	def fwithparams(self,params):
		fs=util.fixparams(self.f,params)
		return util.makeblockwise(fs)

	def compress(self):
		self.f=None
		return self

	def restore(self):
		if 'f' not in vars(self) or self.f==None:
			self.f=self.fdescr.gen_f()
		return self

	def get_type(self):
		return self.fdescr.ftype



class Func(ParameterizedFunc):

	def __init__(self,fdescr):
		super().__init__(fdescr)
		self.weights=self.fdescr.initweights()

	def getclone(self):
		return copy.deepcopy(self)

	def compressedclone(self):
		clone=self.getclone()
		clone.f=None
		return clone

	def eval(self,X):
		return self.fwithparams(self.weights)(X)
	
	
class FunctionDescription:
	def __init__(self,ftype,**kw):
		self.ftype=ftype
		self.kw=kw
		for k,v in kw.items():
			setattr(self,k,v)

	def gen_f(self):
		f=self._gen_f_()
		self.fclass={1:'static',2:'dynamic'}[len(signature(f).parameters)]
		return f if self.fclass=='dynamic' else util.dummyparams(f)

	def _gen_f_(self):
		if self.ftype=='AS_NN':
			NN_NS=mv.gen_NN_NS(self.activation)
			return ASt.gen_Af(self.n,NN_NS)
		if self.ftype=='backflow0':
			return ASt.gen_backflow0(activation=self.activation)
		if self.ftype=='backflow1':
			return ASt.gen_backflow1(activation=self.activation)

		if self.ftype=='singleparticleNN':
			return bf.gen_singleparticleNN(activation=self.activation)


		#static functions
		if self.ftype=='hermiteSlater':
			return examplefunctions.hermiteSlater(self.n,self.d,1/8)
		if self.ftype=='gaussianSlater':
			return examplefunctions.gaussianSlater(self.n,self.d)
		

	def initweights(self):
		return globals()['initweights_{}'.format(self.ftype)](**self.kw) if self.fclass=='dynamic' else None


	def checkwidths(self):
		pass


class ComposedDescription(FunctionDescription):
	def __init__(self,*elements):
		super().__init__('-'.join([e.ftype for e in elements]),elements=elements)

	def _gen_f_(self):
		return util.compose(*[e.gen_f() for e in self.elements])

	def initweights(self):
		return [e.initweights() for e in self.elements]





def get_func(fdescr=None,**kw):
	if fdescr==None: fdescr=FunctionDescription(**kw)
	return Func(fdescr)


def get_composed_func(*elements):
	return Func(ComposedDescription(*elements))


#=======================================================================================================
# antisymmetrized NN
#=======================================================================================================

def initweights_AS_NN(n,d,widths,**kw):
	widths[0]=n*d
	return mv.initweights_NN(widths)


#=======================================================================================================
# backflow+dets
#=======================================================================================================

def initweights_backflow0(n,d,widths,**kw):
	ds,k=widths
	ds[0]=d
	return [bf.initweights_backflow(ds),util.initweights((k,n,ds[-1]))]

def initweights_backflow1(n,d,widths,**kw):
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

def initweights_singleparticleNN(d,widths,**kw):
	widths[0]=d
	return mv.initweights_NN(widths)
