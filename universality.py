import GPU_sum_simple
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import optax
import math
import testing





lossfn=util.sqloss
sumperms=GPU_sum_simple.AS_NN

@jax.jit
def nonsym(Ws,bs,X):
	L1=util.ReLU(util.dot_nd(X,Ws[0])+bs[0][None,:])
	return jnp.inner(L1,Ws[1])


def batchlossAS(Wb,X,Y,lossfn=lossfn):
	W,b=Wb
	Z=sumperms(W,b,X)
	return lossfn(Y,Z)


def lossgradAS(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(batchlossAS)((W,b),X,Y)
	return grad,loss



@jax.jit
def batchlossNS(Wb,X,Y,lossfn=lossfn):
	W,b=Wb
	Z=nonsym(W,b,X)
	loss=lossfn(Y,Z)
	return loss

@jax.jit
def lossgradNS(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(batchlossNS)((W,b),X,Y)
	return grad,loss




def genW(k0,n,d,m=10,randb=False):
	k1,k2,k3=rnd.split(k0,3)
	W0=rnd.normal(k1,(m,n,d))/math.sqrt(n*d)
	W1=rnd.normal(k2,(1,m))/math.sqrt(m)
	W=[W0,W1]
	b=[rnd.normal(k3,(m,))]
	if randb:
		b=[rnd.normal(k0,(m,))]
	return W,b


#
#def features(X):
#	ones=jnp.ones(X.shape[:-1]+(1,))
#	X_=jnp.concatenate([X,ones],axis=-1)
#
#	secondmoments=X_[:,:,:,None]*X_[:,:,None,:]
#	secondmoments=jnp.triu(secondmoments)
#	return jnp.reshape(secondmoments,X_.shape[:-1]+(-1,))


def products(X1,X2):
	
	producttable=X1[:,:,:,None]*X2[:,:,None,:]
	return jnp.reshape(producttable,X1.shape[:-1]+(-1,))


def momentfeatures(k):

	def moments(X):
		ones=jnp.ones(X.shape[:-1]+(1,))
		X_=jnp.concatenate([X,ones],axis=-1)
		Y=X_
		for i in range(k-1):
			Y=products(Y,X_)
		return Y

	return moments
			
secondmoments=momentfeatures(2)


def appendnorm(X):
	sqnorms=jnp.sum(jnp.square(X),axis=-1)
	X_=jnp.concatenate([X,sqnorms],axis=-1)
	return X_
	



features=secondmoments
#features=appendnorm


def nfeatures(n,d,featuremap):
	k=rnd.PRNGKey(0)
	X=rnd.normal(k,(10,n,d))
	out=featuremap(X)
	return out.shape[-1],jnp.var(out)


#single-particle features
class SPfeatures:
	def __init__(self,key,n,d,m,featuremap):
		self.featuremap=featuremap
		d_,var=nfeatures(n,d,featuremap)
		self.W,self.b=genW(key,n,d_,m)
		#self.W,self.b=genW(key,n,d_,m,randb=True)
		self.normalization=1/math.sqrt(var)

		

	def evalblock(self,X):
		F=self.featuremap(X)*self.normalization
		return sumperms(self.W,self.b,F)

	def eval(self,X,blocksize=250):
		samples=X.shape[0]
		blocks=[]
		#blocksize=250
		Yblocks=[]
		a=0
		while a<samples:
			b=min(a+blocksize,samples)
			Yblocks.append(self.evalblock(X[a:b]))
			a=b
		return jnp.concatenate(Yblocks,axis=0)

	def evalNS(self,X):
		F=self.featuremap(X)*self.normalization
		return nonsym(self.W,self.b,F)
		


