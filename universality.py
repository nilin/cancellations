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
import testing




def sqloss(Y,Z):
	Y,Z=[jnp.squeeze(_) for _ in (Y,Z)]
	return jnp.average(jnp.square(Y-Z))

def rtloss(Y,Z):
	return jnp.sqrt(sqloss(Y,Z))

lossfn=sqloss
lossfnNS=sqloss

#def sqnorm(Y):
#	return jnp.average(jnp.square(Y))



def sumperms(W,b,X):
	
	W_=[jnp.expand_dims(L,axis=0) for L in W]
	b_=[jnp.expand_dims(bias,axis=0) for bias in b]
	X_=jnp.reshape(X,(1,1,)+X.shape)
	
	out=GPU_sum.sum_perms_multilayer(W_,b_,X_,'ReLU',mode='silent')
	out=jnp.squeeze(out)

	return out

@jax.jit
def nonsym(Ws,bs,X):
	L1=util.ReLU(util.dot_nd(X,Ws[0])+bs[0][None,:])
	return jnp.inner(L1,Ws[1])


def batchloss(Wb,X,Y,lossfn=lossfn):
	W,b=Wb
	Z=sumperms(W,b,X)
	return lossfn(Y,Z)


def lossgrad(Wb,X,Y):
	W,b=Wb
	loss,grad=jax.value_and_grad(batchloss)((W,b),X,Y)
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



#
#def update(W,b,X,Y):
#	grad,loss=lossgrad(W,b,X,Y)	
#	bk.printbar(loss)
#	for l,dw in enumerate(grad):
#		W[l]=W[l]-.01*dw
#	return W
#	


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
