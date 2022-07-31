
# nilin




import jax
import jax.numpy as jnp
import numpy as np
import pdb


jax.config.update("jax_enable_x64", True)


#@jax.jit
def DETS(A,pivot=False):
	ndets,n=A.shape[:-1]

	dets=jnp.ones((ndets,))
	nonzero=jnp.ones((ndets,))

	for i in range(n):
		if pivot:
			A,signs=ppivot(A,n-i)
			dets=dets*signs

		multiplier=A[:,0,0]
		row=A[:,0,1:]/multiplier[:,None]
		scalings=A[:,1:,0]
		A=A[:,1:,1:]-scalings[:,:,None]*row[:,None,:]
		
		nonzero=jnp.logical_and(nonzero,multiplier!=0)
		dets=dets*multiplier*nonzero

	return dets


def ppivot(A,n):
	i_s=jnp.argmax(A[:,:,0]**2,axis=1)
	swaps,signs=swaps_as_perms(jnp.zeros_like(i_s),i_s,n)
	Ps=(jnp.arange(n)[None,:,None]-swaps[:,None,:])==0
	return jax.vmap(jnp.dot)(Ps,A),signs

def swaps_as_perms(i_s,j_s,n):

	trivial=i_s==j_s; nontrivial=1-trivial
	signs=trivial-nontrivial

	i_s=nontrivial*i_s+trivial*(-1)
	j_s=nontrivial*j_s+trivial*(-1)

	imask=(jnp.arange(n)[None,:]-i_s[:,None])==0
	jmask=(jnp.arange(n)[None,:]-j_s[:,None])==0
	fixedmask=1-imask-jmask

	return imask*j_s[:,None]+jmask*i_s[:,None]+fixedmask*jnp.arange(n)[None,:],signs





if __name__=='__main__':

	import jax.random as rnd


	A=rnd.normal(rnd.PRNGKey(0),(100,10,10))
	print(jnp.linalg.det(A))
	print(DETS(A))
	print(DETS(A,pivot=True))

	v=rnd.normal(rnd.PRNGKey(1),(100,10))
	w=rnd.normal(rnd.PRNGKey(2),(100,10))
	A=v[:,:,None]*v[:,None,:]+w[:,:,None]*w[:,None,:]

	print(jnp.sum(jnp.linalg.det(A)**2))
	print(jnp.sum(DETS(A)))
	print(jnp.sum(DETS(A,pivot=True)))


	v=rnd.normal(rnd.PRNGKey(1),(100,10))
	A=v[:,:,None]*v[:,None,:]

	print(jnp.sum(jnp.linalg.det(A)**2))
	print(jnp.sum(DETS(A)))
	print(jnp.sum(DETS(A,pivot=True)))
