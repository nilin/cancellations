import jax.numpy as jnp
import jax
import math
import itertools
from ..utilities import numutil as mathutil
import numpy as np
import jax.random as rnd
import pdb
from numpy.testing import assert_allclose



def naiveAS(f,X):
	samples,n,d=X.shape
	p0=list(range(n))
	out=0
	for p in itertools.permutations(p0):
		sgn=sign(p)
		PX=X[:,p,:]
		out=out+sgn*f(PX)
	return out


def verify_antisymmetrization(Af,f,X):
	verify_antisymmetric(Af,X,rtol=1/10**5)
	verify_antisymmetric(lambda _:naiveAS(f,_),X,rtol=1/10**5)
	Y1=Af(X)
	Y2=naiveAS(f,X)
	assert_allclose(Y1,Y2,rtol=1/10**5)


def verify_antisymmetric(f,X,nperms=10,**kw):

	n=X.shape[-2]
	perms=[np.random.permutation(n) for _ in range(nperms)]
	signs=[sign(p) for p in perms]
		
	Perms=[perm_as_function(p) for p in perms]
	should_be_equal=[s*f(P(X)) for s,P in zip(signs,Perms)]

	#should_be_equal=[]
	#for i,(s,P) in enumerate(zip(signs,Perms)):
	#	if cfg.trackcurrenttask('verifying antisymmetry',(i+1)/len(signs))=='b': return None
	#	should_be_equal.append(s*f(P(X)))
	assert_ALLclose(should_be_equal,**kw)


def verify_equivariant(F,n,d,samples=25,nperms=25,fixparams=None):

	if fixparams!=None:
		F=mathutil.fixparams(F,fixparams)

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
		return mathutil.apply_on_n(P,X)
	return perm

