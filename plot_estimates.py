import matplotlib.pyplot as plt
import jax.numpy as jnp
import bookkeep as bk
from scipy.io import loadmat

n_=range(3,25)

def getmat(n):
	#data=loadmat('matlab/Ints/ReLU/n='+str(n))
	return bk.get('computed_by_integral/n='+str(n))	

A=jnp.stack([getmat(n) for n in n_],axis=0)
plt.plot(n_,jnp.average(A,axis=1))
plt.yscale('log')
plt.show()
