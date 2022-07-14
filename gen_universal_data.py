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
import sys


def picksamplesize(n):
	return 10**6

def pickYsize(n):
	samplesizes=[10**6]*6+[10**4]+[10**5]+[10**3]
	return int(samplesizes[n])

def pickblocksize(n):
	return int(picksamplesize(n)//100)

k0=rnd.PRNGKey(0)
k1,k2=rnd.split(k0)


nmax=int(sys.argv[1])
print('inputs')

for d in [1,3]:
	print('d='+str(d))

	for n in range(1,nmax+1):

		print(n)
		samples=picksamplesize(n)

		X_train=rnd.normal(k1,(samples,n,d))
		X_test=rnd.normal(k2,(samples,n,d))

		bk.save(X_train,'data/X_train_n='+str(n)+'_d='+str(d))
		bk.save(X_test,'data/X_test_n='+str(n)+'_d='+str(d))

#		if n>1:
#			X_test_swap=jnp.concatenate([X_test[:,1:2,:],X_test[:,0:1,:],X_test[:,2:,:]],axis=1)
#			bk.save(X_test_swap,'data/X_test_swap_n='+str(n)+'_d='+str(d))


print('outputs NS')


for d in [1,3]:
	print('\nd='+str(d))
	for n in range(1,nmax+1):

		samplesize=picksamplesize(n)

		print('n='+str(n))
		for m in {1,10}:
		
			X_train=bk.get('data/X_train_n='+str(n)+'_d='+str(d))

			spf=universality.SPfeatures(k0,n,d,m,universality.features)
			targetNS=lambda X:spf.evalNS(X)

			Z_train=targetNS(X_train)

			print(Z_train.shape)

			bk.save(Z_train,'data/Z_train_n='+str(n)+'_d='+str(d)+'_m='+str(m))

print('outputs')


for d in [1,3]:
	print('\nd='+str(d))
	for n in range(1,nmax+1):

		samplesize=pickYsize(n)
		blocksize=pickblocksize(n)

		print('n='+str(n))
		for m in {1,10}:
		
			X_train=bk.get('data/X_train_n='+str(n)+'_d='+str(d))
			X_train=X_train[:samplesize]
			X_test=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
		

			spf=universality.SPfeatures(k0,n,d,m,universality.features)
			target=lambda X:spf.eval(X,blocksize=blocksize)
			targetNS=lambda X:spf.nonsym(X)

			Y_train=target(X_train)
			Y_test=target(X_test)

			bk.save(Y_train,'data/Y_train_n='+str(n)+'_d='+str(d)+'_m='+str(m))
			bk.save(Y_test,'data/Y_test_n='+str(n)+'_d='+str(d)+'_m='+str(m))

#			if n>1:
#				X_test_swap=bk.get('data/X_test_swap_n='+str(n)+'_d='+str(d))
#				Y_test_swap=target(X_test_swap)
#				bk.save(Y_test_swap,'data/Y_test_swap_n='+str(n)+'_d='+str(d)+'_m='+str(m))

