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



print('args n m d minibatchsize')

n=int(sys.argv[1])
m=int(sys.argv[2])
d=int(sys.argv[3])
minibatchsize=int(sys.argv[4])
	

k0=rnd.PRNGKey(0)

W=universality.genW(k0,n,d,m)


X_train=bk.get('data/X_train_n='+str(n)+'_d='+str(d))
Y_train=bk.get('data/Y_train_n='+str(n)+'_d='+str(d))
samples=X_train.shape[0]

d=X_train.shape[-1]

opt=optax.rmsprop(.01)
state=opt.init(W)

iterations=1000
k10=rnd.PRNGKey(10)
_,*keys=rnd.split(k10,iterations)

def regularize(W):
	return [w*.99 for w in W]

for i,k in enumerate(keys):

	I=rnd.choice(k,samples,(minibatchsize,),replace=False)
	#I=jnp.arange(minibatchsize)



	X=jnp.take(X_train,I,axis=0)
	Y=jnp.take(Y_train,I,axis=0)

	grad,loss=universality.lossgrad(W,X,Y)

	updates,state=opt.update(grad,state,W)
	W=optax.apply_updates(W,updates)
	W=regularize(W)

	rloss=loss/universality.lossfn(Y,0)
	bk.printbar(rloss,rloss)
