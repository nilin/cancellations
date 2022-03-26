import math
import pickle
import bookkeep as bk
import jax
import jax.numpy as jnp
import util
import scratchwork as sc



def Znorm(key,activation,n,_):
	z=jax.random.normal(key,shape=(10000,))
	return util.L2norm(activation(z))

def OPnorm(key,activation,n,data):
	d=data['d']
	x=util.sample_mu(n*d,10000,key)
	return util.L2norm(activation(x))

def polyOPnorm(key,activation,n,data):
	d=data['d']
	x=util.sample_mu(n*d,10000,key)
	a,dist=util.polyfit(x,activation(x),n-2)
	return dist

def polyZnorm(key,activation,n,_):
	z=jax.random.normal(key,shape=(10000,))
	a,dist=util.polyfit(z,activation(z),n-2)
	return dist

def OCPnorm(key,activation,n,data):
	W=data['Ws'][n]
	variances=jnp.sum(jnp.square(W),axis=(-2,-1))
	covariances=variances-jnp.square(util.mindist(W))
	return jnp.sqrt(jnp.average(util.variations(key,activation,variances,covariances)))

def polyOCPnorm(key,activation,n,data):
	W=data['Ws'][n]
	variances=jnp.sum(jnp.square(W),axis=(-2,-1))
	covariances=variances-jnp.square(util.mindist(W))
	_,dist=util.poly_fit_variations(key,activation,n-2,variances,covariances)
	return util.L2norm(dist)
		
def polyOCP_norm(key,activation,n,data):

	W=data['Ws_ordered'][n]
	covs,signs=sc.covs(W,3)
	covs=covs+0.00001*jnp.eye(covs.shape[-1])[None,:,:]

	a,dist=util.poly_fit_generalized_variations(key,activation,n-2,covs,signs)
	return util.L2norm(dist)
	
	
def polyOCP_proxynorm(key,activation,n,data):
	d=data['d']
	W=data['Ws'][n]
	delta=data['deltas'][n]
	key1,key2=jax.random.split(key)
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))

	x=util.sample_mu(n*d,10000,key1)
	a,dist=util.polyfit(x,activation(x),n-2)
	p=util.poly_as_function(a)
	r=lambda x:activation(x)-p(x)
	return 1

def gammanorm(key,activation,n,data):
	d=data['d']
	r=1/jnp.sqrt(n*d)
	log_r=jnp.array([jnp.log(r)])

	dists=data['gamma_dists'][n-2]
	f=util.listasfunction(dists['log_r'],dists['L2dist'])

	return jnp.sqrt(math.factorial(n)*f(log_r))



def gamma_HS_norm(key,activation,n,data):

	gamma=util.gamma_HS

	d=data['d']
	r=1/jnp.sqrt(n*d)
	x=r*jax.random.normal(key,(10000,))
	a,dist=util.polyfit(x,gamma(x),n-2)

	return jnp.sqrt(math.factorial(n)*dist)

"""
activation-specific proxies
"""
def exactexp(W,X):
	n=W.shape[-2]
	nfactor=1.0
	for k in range(1,n+1):
		nfactor=nfactor/jnp.sqrt(k)
	instances_samples_n_n=jnp.swapaxes(jnp.exp(jnp.inner(W,X)),1,2)
	return nfactor*jnp.linalg.det(instances_samples_n_n)

#def tanhtaylor(n_):
#	a=[1.0]
#	N=20
#	for n in range(N):
#		s=0
#		for k in range(n+1):
#			s=s+a[k]*a[n-k]	
#		a.append(s/(2*n+3))
#	return jnp.array([jnp.sqrt(a[math.floor(n/2-1)]*a[math.ceil(n/2-1)]) for n in n_])

def exptaylor(n_):
	return [1/math.factorial(n-1) for n in n_]

def expapprox(n_):
	d=3
	n_=jnp.array(n_)
	log_est=-jnp.multiply((n_-d-1),jnp.log(2*jnp.square(n_)))-d*jnp.log(d)+jnp.log(n_)+1
	return jnp.sqrt(jnp.exp(log_est))

#######################
#
def expapproxnorm(key,activation,n,data):
	return expapprox(jnp.array([n]))
