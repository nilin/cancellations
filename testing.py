import jax.numpy as jnp
import jax
import math
import itertools
import util
import numpy as np
import jax.random as rnd
import pdb
import config as cfg
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)


def naiveAS(f,X):
	samples,n,d=X.shape
	p0=list(range(n))
	out=0
	for p in itertools.permutations(p0):
		sign=sign(p)
		PX=X[:,p,:]
		out=out+sign*f(X)
	return out


def verify_antisymmetrization(Af,f,X):
	Y=Af(X)
	Z=naiveAS(f,X)
	assertequal(Y,Z)


def verify_antisymmetric(f,X,nperms=10):

	n=X.shape[-2]
	perms=[np.random.permutation(n) for _ in range(nperms)]
	signs=[sign(p) for p in perms]
		
	Perms=[perm_as_function(p) for p in perms]
	should_be_equal=[s*f(P(X)) for s,P in zip(signs,Perms)]

	assert_ALLclose(should_be_equal)


def verify_equivariant(F,n,d,samples=25,nperms=25,fixparams=None):

	if fixparams!=None:
		F=util.fixparams(F,fixparams)

	X=rnd.normal(rnd.PRNGKey(0),(samples,n,d))
	perms=[np.random.permutation(n) for _ in range(nperms)]
	invperms=[inv(p) for p in perms]

	Perms=[perm_as_function(p) for p in perms]
	InvPerms=[perm_as_function(p) for p in invperms]

	should_be_equal=[Q(F(P(X))) for P,Q in zip(Perms,InvPerms)]
	assert_ALLclose(should_be_equal)


def assert_ALLclose(arrays,*args,**kw):
	Array=jnp.stack(arrays,axis=0)
	assert_allclose(jnp.min(Array,axis=0),jnp.max(Array,axis=0),*args,**kw)




@jax.jit
def sign(p):
	n=len(p)
	p_=jnp.array(p)
	pi_minus_pj=p_[:,None]-p_[None,:]
	pi_gtrthan_pj=jnp.heaviside(pi_minus_pj,0)
	inversions=jnp.sum(jnp.triu(pi_gtrthan_pj))
	return 1-2*(inversions%2)
	
def inv(p):
	(n,)=p.shape
	q=np.zeros((n,))
	for i,j in enumerate(p):
		q[j]=i
	return q

@jax.jit
def perm_as_matrix(p):
	(n,)=p.shape
	P=(p[:,None]-jnp.arange(n)[None,:])==0
	return P

def perm_as_function(p):
	P=perm_as_matrix(p)
	def perm(X):
		return util.apply_on_n(P,X)
	return perm

