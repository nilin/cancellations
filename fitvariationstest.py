import math
import pickle
import bookkeep as bk
import jax
import jax.numpy as jnp
import util
import matplotlib.pyplot as plt

key=jax.random.PRNGKey(0)
key0,*keys=jax.random.split(key,100)
ac_name=input('activation: ')
f=util.activations[ac_name]

vars_=jnp.ones(10)
covs_=jnp.array(range(10))/10
print(covs_)

A=[]
for n in range(10):
	a,dist=util.poly_fit_variations(keys[n],f,n,vars_,covs_)
	A.append(a)

for n in [9]:
	#F=util.poly_as_function(a)
	a=A[n]
	F=util.polys_as_parallel_functions(a)
	X=jnp.arange(-1,1,.02)
	#Y=F(X[None,:])
	Y=F(jnp.repeat(X[None,:],10,axis=0))
	for i in range(Y.shape[0]):
		plt.plot(X,Y[i])
	plt.plot(X,f(X),'r')
	plt.show()
