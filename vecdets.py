
# nilin




import jax
import jax.numpy as jnp
import numpy as np
import pdb





def DETS(A):
	ndets,n=A.shape[:-1]

	dets=jnp.ones((ndets,))
	nonzero=jnp.ones((ndets,))

	for i in range(n):
		multiplier=A[:,0,0]
		row=A[:,0,1:]/multiplier[:,None]
		scalings=A[:,1:,0]
		A=A[:,1:,1:]-scalings[:,:,None]*row[:,None,:]
		
		nonzero=jnp.logical_and(nonzero,multiplier!=0)
		dets=dets*multiplier*nonzero

	return dets


if __name__=='__main__':


	A=jnp.array(np.random.rand(10,5,5))
	print(jnp.linalg.det(A))
	print(DETS(A))

	v=jnp.array(np.random.rand(10,5))
	A=v[:,:,None]*v[:,None,:]

	print(jnp.linalg.det(A))
	print(DETS(A))
