import GPU_sum
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import GPU_sum
import optax
import math





def sqloss(Y,Z):
	return jnp.average(jnp.square(Y-Z))

def rtloss(Y,Z):
	return jnp.sqrt(sqloss(Y,Z))

lossfn=sqloss

#def sqnorm(Y):
#	return jnp.average(jnp.square(Y))


def sumperms(W,X):
	
	W_=[jnp.expand_dims(L,axis=0) for L in W]
	X_=jnp.reshape(X,(1,1,)+X.shape)
	
	out=GPU_sum.sum_perms_multilayer(W_,X_,'ReLU',mode='silent')
	out=jnp.squeeze(out)

	return out


def batchloss(W,X,Y,lossfn=lossfn):
	Z=sumperms(W,X)
	return lossfn(Y,Z)


def lossgrad(W,X,Y):
	loss,grad=jax.value_and_grad(batchloss)(W,X,Y)
	return grad,loss


def update(W,X,Y):
	grad,loss=lossgrad(W,X,Y)	
	bk.printbar(loss)
	for l,dw in enumerate(grad):
		W[l]=W[l]-.01*dw
	return W
	


def genW(k0,n,d,m=10):
	k1,k2=rnd.split(k0)
	W0=rnd.normal(k1,(m,n,d))/math.sqrt(n*d)
	W1=rnd.normal(k2,(1,m))/math.sqrt(m)
	W=[W0,W1]
	return W


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
		self.W=genW(key,n,d_,m)
		#self.normalization=1/math.sqrt(var)
		self.normalization=1
		

	def evalblock(self,X):
		F=self.featuremap(X)*self.normalization
		return sumperms(self.W,F)

	def eval(self,X):
		samples=X.shape[0]
		blocks=[]
		blocksize=250
		Yblocks=[]
		a=0
		while a<samples:
			b=min(a+blocksize,samples)
			Yblocks.append(self.evalblock(X[a:b]))
			a=b
		return jnp.concatenate(Yblocks,axis=0)





#	
#
#k0=rnd.PRNGKey(0)
#k1,k2=rnd.split(k0)
#
#n=8
#d=3
#m=n*d
#
#W=genW(k2,d,m=20)
#
##W_target=genW(k1,10)
##target=lambda X:sumperms(W_target,X)
#
#spf=SPfeatures(k1,d,2,features)
#target=lambda X:spf.eval(X)
#
#samples=10000
#minibatchsize=100
#
#X_=rnd.normal(k0,(samples,n,d))
#
#
#opt=optax.rmsprop(.01)
#state=opt.init(W)
#
#iterations=1000
#k10=rnd.PRNGKey(10)
#_,*keys=rnd.split(k10,iterations)
#
#losses=[]
#
#for i,k in enumerate(keys):
#
#	X=rnd.choice(k,X_,(minibatchsize,),replace=False)
#
#	Y=target(X)
#	grad,loss=lossgrad(W,X,Y)
#	losses.append(loss)
#
#	updates,state=opt.update(grad,state)
#	W=optax.apply_updates(W,updates)
#
#	rloss=loss/sqnorm(Y)
#	#rloss=loss/losses[0]
#	bk.printbar(rloss,rloss)
