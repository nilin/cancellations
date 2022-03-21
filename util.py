import numpy as np
import math
import matplotlib
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import optax
	

pwr=lambda x,p:jnp.power(x,p*jnp.ones(x.shape))

ReLU=lambda x:(jnp.abs(x)+x)/2
#ReLU=lambda x:jnp.max(x,jnp.zeros(x.shape))
DReLU=lambda x:(jnp.abs(x+1)-jnp.abs(x-1))/2
heaviside=lambda x:jnp.heaviside(x,1)
osc=lambda x:jnp.sin(100*x)
softplus=lambda x:jnp.log(jnp.exp(x)+1)

#activations={'softplus':softplus,'osc':osc,'HS':heaviside,'ReLU':ReLU,'exp':jnp.exp,'tanh':jnp.tanh,'DReLU':DReLU}
activations={'exp':jnp.exp,'HS':heaviside,'ReLU':ReLU,'tanh':jnp.tanh,'softplus':softplus,'DReLU':DReLU,'osc':osc}


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
	energies=jnp.triu(1/pairwisedists(X),k=1)
	return 1/jnp.max(energies,axis=(-2,-1))
	

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


def correlated_X_pairs(key,marginal_var,diff_var,samples=250):
	key1,key2=jax.random.split(key)
	r_=jnp.sqrt(marginal_var-diff_var/4)
	eps=jnp.sqrt(diff_var)
	instances=r_.size
	Z=jax.vmap(jnp.multiply,in_axes=(0,0))(jax.random.normal(key1,shape=(instances,samples)),r_)
	Z_=jax.vmap(jnp.multiply,in_axes=(0,0))(jax.random.normal(key2,shape=(instances,samples)),eps)

	X1=Z-Z_/2
	X2=Z+Z_/2
	return X1,X2
	

def variations(key,f,marginal_var,diff_var):
	X1,X2=correlated_X_pairs(key,marginal_var,diff_var)
	return jnp.average(jnp.square(f(X2)-f(X1)),axis=-1)/2


def fit_variations(key,f,functions,marginal_var,diff_var):
	X1,X2=correlated_X_pairs(key,marginal_var,diff_var)
	ydiffs=(f(X2)-f(X1))/jnp.sqrt(2)
	basisdiffs=(functions(X2)-functions(X1))/jnp.sqrt(2)
	return jax.vmap(basisfit,in_axes=(0,0),out_axes=(0,0))(ydiffs,basisdiffs) 
	
def poly_fit_variations(key,f,deg,marginal_var,diff_var):
	functions=monomials(deg)
	return fit_variations(key,f,functions,marginal_var,diff_var)


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
	
####################################################################################################


def monomials(deg,kmin=0):
	def functions(x):
		y=jnp.ones(x.shape)
		vals=[]
		for k in range(kmin,deg+1):
			vals.append(jnp.expand_dims(y,axis=-1))
			y=jnp.multiply(x,y)
		return jnp.concatenate(vals,axis=-1)
	return functions


def basisfit(y,Y):
	Q,R=jnp.linalg.qr(Y)
	Py=jnp.dot(Q.T,y)
	#a=jnp.linalg.multi_dot([jnp.linalg.inv(R),Q.T,y])
	a=jnp.dot(jnp.linalg.inv(R),Py)
	dist=L2norm(y-jnp.dot(Q,Py))
	return a,dist

def functionfit(x,y,functions):
	#Y=jax.vmap(functions,out_axes=0)(x)
	Y=functions(x)
	return basisfit(y,Y)

def functionlistfit(x,y,functionlist):
	return functionfit(x,y,prepfunctions(functionlist))
	
	
def polyfit(x,y,deg):
	return functionfit(x,y,monomials(deg))

def prepfunctions(functionblocklist,functionlist):
	def functions(x):
		return jnp.concatenate([f(x) for f in functionblocklist]+[jnp.expand_dims(f(x),axis=-1) for f in functionlist], axis=-1)
	return functions



####################################################################################################


def compare(x,y):
	rel_err=jnp.linalg.norm(y-x,axis=-1)/jnp.linalg.norm(x,axis=-1)
	print('maximum relative error')
	print(jnp.max(rel_err))
	print()



def normalize(W):
	norms=jnp.sqrt(jnp.sum(jnp.square(W),axis=(-2,-1)))
	return jax.vmap(jnp.multiply,in_axes=(0,0))(W,1/norms)



