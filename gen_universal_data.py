import GPU_sum
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import GPU_sum
import optax
import math
import universality


k0=rnd.PRNGKey(0)
k1,k2=rnd.split(k0)

samples=1000

for d in range(1,4):
	for n in range(1,9):

		print(n)

		X_train=rnd.normal(k1,(samples,n,d))
		X_test=rnd.normal(k2,(1000,n,d))

		bk.save(X_train,'data/X_train_n='+str(n)+'_d='+str(d))
		bk.save(X_test,'data/X_test_n='+str(n)+'_d='+str(d))


		spf=universality.SPfeatures(k0,n,d,1,universality.features)
		target=lambda X:spf.eval(X)

		Y_train=target(X_train)
		Y_test=target(X_test)

		bk.save(Y_train,'data/Y_train_n='+str(n)+'_d='+str(d))
		bk.save(Y_test,'data/Y_test_n='+str(n)+'_d='+str(d))

