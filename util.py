import numpy as np
import math
import bookkeep as bk
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import permutations
	

Wtypes={'s':'separated','n':'normal','ss':'separated small','ns':'normal small','nl':'normal large','sl':'separated large'}
pwr=lambda x,p:jnp.power(x,p*jnp.ones(x.shape))

ReLU=lambda x:(jnp.abs(x)+x)/2
DReLU=lambda x:(jnp.abs(x+1)-jnp.abs(x-1))/2
heaviside=lambda x:jnp.heaviside(x,1)
osc=lambda x:jnp.sin(100*x)
softplus=lambda x:jnp.log(jnp.exp(x)+1)
ac_test=lambda x:ReLU(x)*jnp.cos(100*x)
#ac_test=lambda x:ReLU(x)*jnp.sin(x)


gamma_ReLU=lambda T,t:1/(2*math.pi)*jnp.sqrt(jnp.square(T)-jnp.square(t))+t/math.pi*jnp.arctan(jnp.sqrt((T+t)/(T-t)))
gamma_HS=lambda T,t:jnp.arctan(jnp.sqrt((T+t)/(T-t)))/math.pi

def gen_gamma_ReLU(Ts):
	return lambda t:jax.vmap(gamma_ReLU,in_axes=(0,0))(Ts*1.001,t)
def gen_gamma_HS(Ts):
	return lambda t:jax.vmap(gamma_ReLU,in_axes=(0,0))(Ts*1.001,t)




#activations={'exp':jnp.exp,'HS':heaviside,'ReLU':ReLU,'tanh':jnp.tanh,'softplus':softplus,'DReLU':DReLU,'osc':osc}
activations={'exp':jnp.exp,'HS':heaviside,'ReLU':ReLU,'tanh':jnp.tanh,'osc':osc}


L2norm=lambda y:jnp.sqrt(jnp.average(jnp.square(y)))
L2over=lambda y,**kwargs:jnp.sqrt(jnp.average(jnp.square(y),**kwargs))


def flatten_nd(x):
	s=x.shape
	newshape=s[:-2]+(s[-2]*s[-1],)
	return jnp.reshape(x,newshape)

def separate_n_d(x,n,d):
	s=x.shape
	newshape=s[:-1]+(n,d)
	return jnp.reshape(x,newshape)
	

def vmapslice(f,*arrays,b):
	mapshape=arrays[0].shape[:b]
	mapsize=jnp.product(jnp.array(mapshape))
	arrays_=[jnp.reshape(A,(mapsize,)+A.shape[b:])]
	mapped=jax.vmap(f,*arrays_)
	return jnp.reshape(mapped,mapshape+mapped.shape[1:])


def pairwisediffs(X):
	n=X.shape[-2]
	stacked_x_1=jnp.repeat(jnp.expand_dims(X,-2),n,axis=-2)
	stacked_x_2=jnp.swapaxes(stacked_x_1,-2,-3)
	return stacked_x_1-stacked_x_2

def pairwisesquaredists(X):
	return jnp.sum(jnp.square(pairwisediffs(X)),axis=-1)

def pairwisedists(X):
	return jnp.sqrt(pairwisesquaredists(X))

def Coulomb(X):
	energies=jnp.triu(1/pairwisedists(X),k=1)
	return jnp.sum(energies,axis=(-2,-1))


def mindist(X):
	energies=jnp.triu(jnp.exp(-pairwisedists(X)),k=1)
	return -jnp.log(jnp.max(energies,axis=(-2,-1)))
	
def mindist_per_i(X):
	triu=jnp.triu(jnp.exp(-pairwisedists(X)),k=1)
	energies=triu+jnp.swapaxes(triu,-2,-1)
	return -jnp.log(jnp.max(energies,axis=(-1)))

def argmindist(X):
	energies=jnp.triu(1/pairwisedists(X),k=1)
	n=energies.shape[-2]
	E=jnp.reshape(energies,energies.shape[:-2]+(n**2,))
	ijflat=jnp.argmax(E,axis=-1)
	i,j=ijflat//n,ijflat%n
	ij=jnp.moveaxis(jnp.array([i,j]),0,-1)
	return ij


def transposition(x,ij):
	n=x.shape[-2]
	ij=ij.astype(int)
	i,j=ij[0],ij[1]
	permutation=list(range(n))
	permutation[i],permutation[j]=j,i
	return jnp.take(x,jnp.array(permutation,int),axis=-2)
	
	
def transpositions(X,ijs):
	return jax.vmap(transposition,in_axes=(0,0))(X,ijs)



def sample_mu(n,samples,key):
	Z=jax.random.normal(key,shape=(samples,n,2))
	P=jnp.squeeze(jnp.product(Z,axis=-1))
	return jnp.sum(P,axis=-1)/jnp.sqrt(n)

def correlated_X_pairs(key,variances,covariances,samples=10000):
	a,b=variances,covariances
	cov=jnp.moveaxis(jnp.array([[a,b],[b,a]]),-1,0)
	X=normal(key,samples,cov)
	return jnp.take(X,0,axis=-1),jnp.take(X,1,axis=-1)

def normal(key,samples,covs):
	n=covs.shape[-1]
	instances=covs.shape[:-2]
	L=jnp.linalg.cholesky(covs)
	Z=jax.random.normal(key,shape=(instances+(n,samples)))
	return jnp.swapaxes(jax.lax.batch_matmul(L,Z),-1,-2)
	
"""
	key1,key2=jax.random.split(key)
	r_=jnp.sqrt(marginal_var-diff_var/4)
	eps=jnp.sqrt(diff_var)
	instances=r_.size
	Z=jax.vmap(jnp.multiply,in_axes=(0,0))(jax.random.normal(key1,shape=(instances,samples)),r_)
	Z_=jax.vmap(jnp.multiply,in_axes=(0,0))(jax.random.normal(key2,shape=(instances,samples)),eps)

	X1=Z-Z_/2
	X2=Z+Z_/2
	return X1,X2
"""	


def fit_variations(key,f,functions,variances,covariances):
	X1,X2=correlated_X_pairs(key,variances,covariances)
	y=(f(X2)-f(X1))/jnp.sqrt(2)
	basis=(functions(X2)-functions(X1))/jnp.sqrt(2)
	return jax.vmap(basisfit,in_axes=(0,0),out_axes=(0,0))(y,basis) 

def variations(key,f,variances,covariances):
	X1,X2=correlated_X_pairs(key,variances,covariances)
	return jnp.average(jnp.square(f(X2)-f(X1)),axis=-1)/2

	
def poly_fit_variations(key,f,deg,variances,covariances):
	key1,key2=jax.random.split(key)
	coefficients_,dist=fit_variations(key1,f,monomials(deg),variances,covariances)
	coefficients=jnp.concatenate([means(key2,f,variances)[:,None],coefficients_[:,1:]],axis=-1)
	return coefficients,dist

def means(key,f,variances):
	X=jnp.sqrt(variances)[:,None]*jax.random.normal(key,shape=variances.shape+(10000,))
	return jnp.average(f(X),axis=-1)

def as_function(a,functions):
	def function(x):
		return jnp.tensordot(functions(x),a,axes=(-1,-1))
	return function

def as_parallel_functions(a,functionbasis):
	def functions(x):
		Y=functionbasis(x)
		out=jax.vmap(jnp.dot,in_axes=(0,0),out_axes=0)(Y,a)
		return out
	return functions

def poly_as_function(a):
	return as_function(a,monomials(a.shape[-1]-1))

def polys_as_parallel_functions(a):
	return as_parallel_functions(a,monomials(a.shape[-1]-1))




def monomials(deg,kmin=0):
	def functions(x):
		y=jnp.ones(x.shape)
		vals=[]
		for k in range(kmin,deg+1):
			vals.append(jnp.expand_dims(y,axis=-1))
			y=jnp.multiply(x,y)
		return jnp.concatenate(vals,axis=-1)
	return functions

def trigs(degs):
	degs=jnp.array(degs)
	def functions(x):
		theta_x=x[:,None]*degs[None,:]
		return jnp.concatenate([jnp.cos(theta_x),jnp.sin(theta_x)],axis=-1)
	return functions



def basisfit(y,Y):
	Q,R=jnp.linalg.qr(Y)
	Py=jnp.dot(Q.T,y)
	a=jnp.dot(jnp.linalg.inv(R),Py)
	dist=L2norm(y-jnp.dot(Q,Py))
	return a,dist

def functionfit(x,y,functions):
	Y=functions(x)
	return basisfit(y,Y)

def functionlistfit(x,y,functionlist):
	return functionfit(x,y,prepfunctions(functionlist))
	
	
def polyfit(x,y,deg):
	return functionfit(x,y,monomials(deg))


def trigfit(x,y,degs):
	return functionfit(x,y,trigs(degs))

def prepfunctions(functionblocklist,functionlist):
	def functions(x):
		return jnp.concatenate([f(x) for f in functionblocklist]+[jnp.expand_dims(f(x),axis=-1) for f in functionlist], axis=-1)
	return functions

def applymatrixalongdim(M,vectors,axis):
	return jnp.moveaxis(jnp.dot(jnp.moveaxis(vectors,axis,-1),M.T),-1,axis)

####################################################################################################


def compare(x,y):
	rel_err=jnp.linalg.norm(y-x,axis=-1)/jnp.linalg.norm(x,axis=-1)
	print('maximum relative error')
	print(jnp.max(rel_err))
	print()



def assertequal(x,y,msg=''):
	x,y=jnp.array(x),jnp.array(y)
	ratio=x/y
	relerror=jnp.log(ratio)
	ln='\n'+150*'-'+'\n'
	print(ln+'Assert equal: '+msg+'\nvalues\n'+str(jnp.concatenate([jnp.expand_dims(x,-1),jnp.expand_dims(y,-1)],axis=-1)))
	print('\nrel error:\n'+str(relerror))
	assert(jnp.all(ratio>0))
	assert(jnp.max(jnp.abs(relerror))<0.001)
	print('\nTest passed: equal\n'+str(jnp.concatenate([jnp.expand_dims(x,-1),jnp.expand_dims(y,-1)],axis=-1))+ln)
	


def normalize(W):
	norms=jnp.sqrt(jnp.sum(jnp.square(W),axis=(-2,-1)))
	return jax.vmap(jnp.multiply,in_axes=(0,0))(W,1/norms)

def normalize_flex(W):
	norms=jnp.sqrt(jnp.sum(jnp.square(W),axis=(-2,-1)))
	return vmapslice(jnp.multiply,W,1/norms,-2)

####################################################################################################



def generalized_variations(key,fs,cov,signs):
	samples=200
	X=normal(key,samples,cov)	
	y=jax.vmap(jnp.inner,in_axes=(0,0))(signs,fs(X))/jnp.sqrt(cov.shape[-1])
	return jnp.average(jnp.square(y),axis=-1)


def fit_generalized_variations(key,f,functions,cov,signs):
	samples=2000
	X=normal(key,samples,cov)
	scaling=1/jnp.sqrt(cov.shape[-1])
	y=scaling*jnp.inner(signs,f(X))
	basis=scaling*jnp.dot(signs,functions(X))
	return jax.vmap(basisfit,in_axes=(0,0),out_axes=(0,0))(y,basis) 

def poly_fit_generalized_variations(key,f,deg,cov,signs):
	return fit_generalized_variations(key,f,monomials(deg),cov,signs)

#################################################################################################

def smooth(x,y,eps,r=5):
	kernel=jnp.array(list(range(1,r+1))+list(reversed(range(1,r))))/r**2
	x_=x[:-2*r+2]+eps*r
	y_=jnp.convolve(y,kernel,mode='valid')
	return x_,y_


def numdiff(x,y,eps,r=5):
	dkernel=jnp.array(r*[1]+r*[-1])/(r**2*eps)
	dy=jnp.convolve(y,dkernel,mode='valid')
	x_=x[:-2*r+1]+eps*(r-1/2)
	return x_,dy

def extend(x,y,eps,dy=None,pad=25):
	a=x[0]
	b=x[-1]
	I=jnp.arange(-pad,0)*eps+a
	J=jnp.arange(1,pad+1)*eps+b
	x_=jnp.concatenate([I,x,J])

	yI=jnp.array(pad*[y[0]])
	yJ=jnp.array(pad*[y[-1]])

	if dy is not None:
		yJ=yJ+dy[-1]*eps*jnp.arange(1,pad+1)
		yI=yI+dy[0]*eps*jnp.arange(-pad,0)
	y_=jnp.concatenate([yI,y,yJ])
	return x_,y_
	

def listasfunction(x_range,y,fuzziness=.01):

	def delta(x):
		return jnp.exp(-jnp.square(x/fuzziness))
		#return ReLU(-jnp.abs(x)+fuzziness)

	def f(X):
		mask=delta(X[:,None]-x_range[None,:])
		return jnp.inner(mask,y)/jnp.sum(mask,axis=-1)
		
	return f


def medmeans(y_,k):
	if y_.shape[-1]<k:
		return jnp.average(y_,axis=-1)
	n=(y_.shape[-1]//k)*k
	y=jnp.take(y_,jnp.arange(n),axis=-1)
	Y=jnp.reshape(y,y.shape[:-1]+(k,n//k))
	avgs=jnp.average(Y,axis=-1)
	return jnp.median(avgs,axis=-1)

def bootstrap(key,y,resamples=1000):
	n=y.shape[0]
	indices=jax.random.randint(key,(resamples*n,),0,n)
	return jnp.reshape(y[indices],(resamples,n))
	
def bootstrapmeans(key,y,**kwargs):
	samples=bootstrap(key,y,**kwargs)
	return jnp.average(samples,axis=-1)

