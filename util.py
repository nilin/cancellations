import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import pdb
import jax.random as rnd
from jax.lax import collapse	
import config as cfg
import customactivations as ca
from config import session

from jax.nn import softplus













@jax.jit
def sqloss(Y1,Y2):
	Y1,Y2=[jnp.squeeze(_) for _ in (Y1,Y2)]
	return jnp.average(jnp.square(Y1-Y2))


@jax.jit
def dot(Y1,Y2):
	#Y1,Y2=[jnp.squeeze(_) for _ in (Y1,Y2)]
	Y1,Y2=[jnp.atleast_1d(jnp.squeeze(_)) for _ in (Y1,Y2)]
	n=Y1.shape[0]
	return jnp.dot(Y1,Y2)/n


@jax.jit
def SI_loss(Y,Y_target):
	return 1-dot(Y,Y_target)**2/(dot(Y,Y)*dot(Y_target,Y_target))

@jax.jit
def log_SI_loss(Y,Y_target):
	Y,Y_target=[jnp.squeeze(_) for _ in (Y,Y_target)]
	return jnp.log(dot(Y_target,Y_target))+jnp.log(dot(Y,Y))-2*jnp.log(dot(Y,Y_target))




@jax.jit
def prod(L):
	out=1
	for array in L:
		out*=array
	return out


def swap(x,y):
	return (y,x)






@jax.jit
def ReLU(x):
	return jnp.maximum(x,0) 

@jax.jit
def DReLU(x):
	return jnp.minimum(jnp.maximum(x,-1),1)

@jax.jit
def leaky_ReLU(x):
	return jnp.maximum(x,.01*x)


activations={'ReLU':ReLU,'relu':ReLU,'tanh':jnp.tanh,'softplus':softplus,'DReLU':DReLU,"drelu":DReLU,"leakyrelu":leaky_ReLU}|ca.c_acs



@jax.jit
def sqlossindividual(Y1,Y2):
	Y1,Y2=[jnp.squeeze(_) for _ in (Y1,Y2)]
	return jnp.square(Y1-Y2)


@jax.jit
def norm(Y):
	return jnp.sqrt(sqloss(0,Y))


@jax.jit
def relloss(Y1,Y2):
	return sqloss(Y1,Y2)/sqloss(0,Y2)


@jax.jit
def dot_nd(A,B):
	return jnp.tensordot(A,B,axes=([-2,-1],[-2,-1]))



@jax.jit
def collapselast(A,k):
	dims=A.shape
	#return collapse(A,dims-k,dims)
	return jnp.reshape(A,dims[:-2]+(dims[-2]*dims[-1],))


def randperm(*Xs):
	X=Xs[0]
	n=X.shape[0]
	p=np.random.permutation(n)
	PXs=[np.array(X)[p] for X in Xs]
	#return [jnp.stack([Y[p_i] for p_i in p]) for Y in args]
	return [jnp.array(PX) for PX in PXs]
	

@jax.jit
def apply_on_n(A,X):

	_=jnp.dot(A,X)
	out= jnp.swapaxes(_,len(A.shape)-2,-2)

	return out


@jax.jit
def flatten_first(X):
	blocksize=X.shape[0]*X.shape[1]
	shape=X.shape[2:]
	return jnp.reshape(X,(blocksize,)+shape)
	

	


@jax.jit
def allmatrixproducts(As,Bs):
	products=apply_on_n(As,Bs)
	return flatten_first(products)


def scale(f,C):
	#return jax.jit(lambda X:C*f(X))
	return lambda X:C*f(X)


def normalize(f,X_,echo=False):

	scalesquared=sqloss(f(X_),0)
	C=1/math.sqrt(scalesquared)
	if echo:
		cfg.log('normalized by factor {:.3}'.format(C))
	return scale(f,C)


def normalize_by_weights(learner,X_):
	f=learner.as_static()	
	scalesquared=sqloss(f(X_),0)
	C=1/math.sqrt(scalesquared)

	weights=learner.weights
	weights[0][-1]=weights[0][-1]*C



def closest_multiple(f,X,Y_target,normalized=False):
	Y=f(X)
	if normalized:
		C=jnp.sign(jnp.dot(Y,Y_target))/jnp.sqrt(dot(Y,Y))
	else:
		C=jnp.dot(Y,Y_target)/jnp.dot(Y,Y)
	return scale(f,C)





def chop(*Xs,chunksize):
	S=Xs[0].shape[0]
	limits=[(a,min(a+chunksize,S)) for a in range(0,S,chunksize)]
	return [tuple([X[a:b] for X in Xs]) for a,b in limits]
	



def makeblockwise(f,*args):

	loud = 'loud' in args


	def blockwise_f(X):
		_,n,_=X.shape	
		Xs=chop(X,chunksize=cfg.memorybatchlimit(n))
		out=[]
		for i,(B,) in enumerate(Xs):
			out.append(f(B))
			if loud:
				cfg.trackcurrenttask('blockwise eval',(i+1)/len(Xs))
		return jnp.concatenate(out,axis=0)

	return blockwise_f





def addgrads(G1,G2):
	if G1==None:
		return G2
	elif type(G2)==list:
		return [addgrads(g1,g2) for g1,g2 in zip(G1,G2)]
	else:
		return G1+G2
		
def scalegrad(G,r):
	if type(G)==list:
		return [scalegrad(g,r) for g in G]
	else:
		return r*G


def sumgrads(Gs):
	Gsum=None
	for G in Gs:
		Gsum=addgrads(Gsum,G)
	return Gsum

def avg_grads(Gs):
	Gsum=None
	for G in Gs:
		Gsum=addgrads(Gsum,G)
	return scalegrad(Gsum,1/len(Gs))



def distinguishable(x,y,p_val=.10,**kwargs): # alternative='greater' to stop when no longer decreasing
	u,p=st.mannwhitneyu(x,y,**kwargs)
	return p<p_val



def donothing(*args):
	pass


def fixparams(f_,params):

	@jax.jit
	def f(X):
		return f_(params,X)
	return f


def noparams(f_):
	return fixparams(f_,None)


def dummyparams(f):
	@jax.jit
	def f_(_,x):
		return f(x)
	return f_


def keyfromstr(s):
	return rnd.PRNGKey(hash(s))



def applyonleaves(T,fn):
	if type(T)==list or type(T)==tuple:
		return [nestedstructure(e,fn) for e in T]
	else:
		return fn(T)

nestedstructure=applyonleaves


def dimlist(T):
	return nestedstructure(T,lambda A:A.shape)



def applyalonglast(f,X,last):
	lshape,rshape=X.shape[:-last],X.shape[-last:]
	batchsize=np.product(lshape)
	X_=jnp.reshape(X,(batchsize,)+rshape)
	Y_=f(X_)
	return jnp.squeeze(jnp.reshape(Y_,lshape+(-1,)))




#	if type(l)==list:
#		return [dimlist(e) for e in l]
#	else:
#		return l.shape
	
def shapestr(l):
	return str(dimlist(l))


def printshape(l):
	print(shapestr(l))


def scalarfunction(f):
	def g(*inputs):
		return jnp.squeeze(f(*inputs))
	return g


def combinelossgradfns(lossgradfns,nums_inputs,coefficients):
	#@jax.jit
	def combinedlossgradfn(params,X,*Ys):
		losses,grads=zip(*[lossgrad(params,X,*Ys[:numinputs-1]) for lossgrad,numinputs in zip(lossgradfns,nums_inputs)])
		
		total_loss=sum([loss*c for loss,c in zip(losses,coefficients)])
		total_grad=sumgrads([scalegrad(grad,c) for grad,c in zip(grads,coefficients)])
		return total_loss,total_grad

	return combinedlossgradfn




#def deltasquared(w):
#	sqdists=jnp.sum(jnp.square(w[:,None,:]-w[None,:,:]),axis=-1)
#	return 1/jnp.max(jnp.triu(1/sqdists))

def initweights(shape):
	return rnd.normal(cfg.nextkey(),shape)/jnp.sqrt(shape[-1])






def compose(*functions):

	def composed(params,X):
		for f,param in zip(functions,params):
			X=f(param,X)
		return X

	return jax.jit(composed)

	
def recompose(ffff,h):

	def hffff(params,X):
		Y=ffff(params[:-1],X)
		return h(params[-1],Y)

	return hffff




