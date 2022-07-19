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






class Slater():
	def __init__(self,F):			# F:x->(f1(x),..,fn(x))		s,d |-> s,n
		self.F=F

	def AS(X):					# X:	s,n,d
		F=self.F
		FX=jax.vmap(F,in_axes=1,out_axes=-1)(X)	# FX:	s,n (basisfunction),n (particle)
		return jnp.linalg.det(FX)
	


def genhermitefunctions(n):
	coefficients=hermitecoefficientblock(n)
	return genpolynomialfunctions(coefficients)	


class HermiteSlater(Slater):

	def __init__(self,n):
		super().__init__(genhermitefunctions(n))



#----------------------------------------------------------------------------------------------------
# polynomials
#----------------------------------------------------------------------------------------------------



def genmonomialfunctions(n):
	@jax.jit
	def F(x):
		x=jnp.squeeze(x)
		xk=jnp.ones_like(x)
		out=[]
		for k in range(n+1):
			out.append(xk)	
			xk=x*xk
		return jnp.stack(out,axis=-1)
	return F
		

def genpolynomialfunctions(coefficients):	#coefficients dimensions: function,degree
	degree=coefficients.shape[-1]-1
	monos=genmonomialfunctions(degree)
	@jax.jit
	def P(x):
		return jnp.inner(monos(x),coefficients)
	return P
	
	

#----------------------------------------------------------------------------------------------------
# Hermite polynomials
#----------------------------------------------------------------------------------------------------

			
def hermitecoefficients(n):
	if n==0:
		return [jnp.array([1])]
	if n==1:
		return [jnp.array([1]),jnp.array([0,1])]
	else:
		A=hermitecoefficients(n-1)

		#a2=jnp.zeros((2,)) if n==2 else jnp.concatenate([A[-2],jnp.zeros((2,))])
		a2=jnp.concatenate([A[-2],jnp.zeros((2,))])
		a=jnp.concatenate([jnp.zeros((1,)),A[-1]])-(n-1)*a2

		A.append(a)
		return A


def hermitecoefficientblock(n):
	return jnp.stack([jnp.concatenate([p,jnp.zeros((n+1-p.shape[0],))]) for p in hermitecoefficients(n)],axis=0)






#----------------------------------------------------------------------------------------------------
# other target functions
#----------------------------------------------------------------------------------------------------



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
		







#---------------------------------------------------------------------------------------------------- 
# helper functions etc
#---------------------------------------------------------------------------------------------------- 

#def printpolys(P):
#	for p in P:
#		printpoly(p)
#def printpoly(p):
#	n=p.shape[0]-1
#	pstr=' + '.join([str(p[k])+('' if k==0 else 'x' if k==1 else 'x^'+str(k)) for k in range(n,-1,-1) if p[k]!=0])
#	print(pstr)
#printpolys(hermitecoefficients(10))
#print(round(hermitecoefficientblock(10)))





#---------------------------------------------------------------------------------------------------- 
# tests
#---------------------------------------------------------------------------------------------------- 


def plothermites(n):
	x=jnp.arange(-4,4,.01)
	X=jnp.expand_dims(x,axis=-1)

	import matplotlib.pyplot as plt

	F=genhermitefunctions(n)
	Y=F(x)

	for k in range(n):
		plt.plot(x,Y[:,k])
	plt.ylim(-10,10)
	plt.show()

#plothermites(6)
