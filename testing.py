import jax.numpy as jnp
import jax
import math
import itertools
import util
import numpy as np
import jax.random as rnd
import pdb
import config as cfg



def assertequal(y,z,blockdim=0,eps=.001):
	cfg.log('asserting equality')
	loss=util.relloss(y,z)
	cfg.log(loss)
	try:
		assert(loss<eps)
		cfg.log('yes, they agree')
	except:
		cfg.log('error x=/=y')
		cfg.log(jnp.stack([y,z],axis=-blockdim-1))
		raise ValueError



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


def verify_antisymmetric(AS,n,d):
	X=rnd.normal(rnd.PRNGKey(0),(100,n,d))

	Y=AS(X)
	for _ in range(25):
		p=np.random.permutation(n)

		PX=np.array(X)[:,p,:]
		assertequal(AS(PX),Y*sign(p))


@jax.jit
def sign(p):
	n=len(p)
	p_=jnp.array(p)
	pi_minus_pj=p_[:,None]-p_[None,:]
	pi_gtrthan_pj=jnp.heaviside(pi_minus_pj,0)
	inversions=jnp.sum(jnp.triu(pi_gtrthan_pj))
	return 1-2*(inversions%2)
	

