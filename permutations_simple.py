# nilin

import numpy as np
import math
import pickle
import time
import copy
import bookkeep as bk
import util
import jax
import sys
import jax.numpy as jnp
import bisect
import time
import testing




def toplevelperm(n,k):
	I=jnp.eye(n)
	top=I[1:k+1]
	mid=I[0:1]
	bottom=I[k+1:]
	return jnp.concatenate([top,mid,bottom],axis=0)
	

def toplevelperms(n):
	perms=jnp.stack([toplevelperm(n,k) for k in range(n)],axis=0)
	signs=(-1)**jnp.arange(n)
	return perms,signs


def permblocks(n,level):
	Ps,signs=toplevelperms(level)
	h,r=level,n-level
	top=jnp.eye(r,n)
	tops=jnp.broadcast_to(top,(h,r,n))
	#tops=top[None,:,:]
	bottoms=jnp.concatenate([jnp.zeros((h,level,r)),Ps],axis=-1)
	return jnp.concatenate([tops,bottoms],axis=-2),signs

def allperms(n):
	allPs,allsigns=permblocks(n,1)
	for level in range(2,n+1):
		Ps,signs=permblocks(n,level)
		allPs=util.allmatrixproducts(Ps,allPs)
		allsigns=jnp.ravel(signs[:,None]*allsigns[None,:])
	return jnp.array(allPs,dtype=int),allsigns
	


def allpermtuples(n):
	Ps,signs=allperms(n)
	return permtuple(Ps),signs


def permtuple(Ps):
	n=Ps.shape[-1]
	return jnp.dot(jnp.arange(n),Ps)

	

"""
----------------------------------------------------------------------------------------------------

def toplevelperm(n,k):
	return jnp.concatenate([k,jnp.arange(k),jnp.arange(k+1,n)])

def toplevelperms(n):
	perms=jnp.stack([toplevelperm(n,k) for k in range(n)],axis=0)
	signs=(-1)**jnp.arange(n)
	return perms,signs

def permblocks(n,level):
	perms_,signs=toplevelperms(level)
	r=n-level
	left=jnp.zeros((level,))[:,None]+jnp.arange(r)[None,:]
	perms=jnp.concatenate([left,perms_+r],axis=0)
	return perms,signs

----------------------------------------------------------------------------------------------------
"""





"""

----------------------------------------------------------------------------------------------------
test
----------------------------------------------------------------------------------------------------
"""

def testallperms(n):
	P,s=allperms(n)
	testing.testperms(P,s)


if __name__=='__main__':
	testallperms(5)
