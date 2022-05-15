import jax
import jax.numpy as jnp
import GPU_sum
import bookkeep as bk
import math
import testing
import util
import sys


def split_data(Xs):
	samples,n,d=Xs.shape
	batchsize=max(round(100000/math.factorial(n)),1)
	start=0
	batches=[]
	while start<samples:
		end=start+min(batchsize,samples)
		batches.append(Xs[start:end])
		start=end
	return batches



nmax=int(sys.argv[1])
depth=int(sys.argv[2])
ac_name=sys.argv[3]
scaling=sys.argv[4]
instances=int(sys.argv[5])
samples=int(sys.argv[6])

#nmax=int(input('nmax: '))
#depth=int(input('depth: '))
#ac_name='tanh'
#scaling='X'
#ac_name=input('activation: ')
#instances=5
#samples=10

for n in range(2,nmax+1):
	Ws=bk.get('inputs/Ws/n='+str(n)+' depth='+str(depth)+' scaling='+scaling)
	Xs=bk.get('inputs/Xs/n='+str(n))
	
	Ws=Ws[:min(len(Ws),instances)]
	Xs=Xs[:min(Xs.shape[0],samples)]

	AS=[]
	for i,W in enumerate(Ws):
		print('instance '+str(i)+100*'-')
		Xs_=split_data(Xs)
		AS.append(GPU_sum.sum_perms_multilayer(W,Xs_,ac_name))

	fn=ac_name+' n='+str(n)+' depth='+str(depth)+' scaling='+scaling
	bk.save(jnp.stack(AS,axis=0),'outputs/AS '+fn)


	NS=jnp.stack([testing.NN_nd(W,Xs) for W in Ws],axis=0)
	bk.save(NS,'outputs/NS '+fn)

