# nilin

#----------------------------------------------------------------------------------------------------
# This file replaces permutations
#----------------------------------------------------------------------------------------------------

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





def applyrightshift_to_rows(shift,A):
	i,j=shift
	return jnp.concatenate([A[:,:i,:],A[:,i+1:j+1,:],A[:,i:i+1,:],A[:,j+1:,:]],axis=-2)



def rightshifts_of_i(i,n):
	shifts=[(i,j) for j in range(i,n)]
	signs=(-1)**jnp.arange(len(shifts))
	return shifts,signs


def allpermsof(x,fix_first=0,keep_order_of_last=0):
	n=x.shape[-2]
	Ps=jnp.expand_dims(x,0)
	allsigns=jnp.array([1])

	for i in range(n-1-keep_order_of_last,fix_first-1,-1):	
		shifts,shiftsigns=rightshifts_of_i(i,n)
		Ps=jnp.concatenate([applyrightshift_to_rows(S,Ps) for S in shifts],axis=0)
		allsigns=jnp.ravel(shiftsigns[:,None]*allsigns[None,:])

	return Ps,allsigns

def allperms(n,**kwargs):
	I=jnp.eye(n)
	return allpermsof(I,**kwargs)

def allpermtuples(n,**kwargs):
	I=jnp.array([[i] for i in range(n)])
	P,s=allpermsof(I)
	return jnp.squeeze(P),s

#----------------------------------------------------------------------------------------------------


# testing ----------------------------------------------------------------------------------------------------




if __name__=='__main__':
	print(allpermtuples(3))
	print(allperms(3))

	

