import jax.numpy as jnp
import jax
import math
import itertools
import util
import GPU_sum




def NN(Ws,X,ac='ReLU'):
	activation={'ReLU':util.ReLU,'tanh':jnp.tanh,'HS':util.heaviside,'DReLU':util.DReLU}[ac]
	X=X.T
	for W in Ws[:-1]:
		X=activation(jnp.dot(W,X))
	return jnp.dot(Ws[-1],X)


def NN_nd(Ws,X,**kwargs):
	n,d=X.shape[-2:]
	flatW=jnp.reshape(Ws[0],Ws[0].shape[:-2]+(n*d,))
	flatX=jnp.reshape(X,X.shape[:-2]+(n*d,))
	Ws_=[flatW]+Ws[1:]

	return NN(Ws_,flatX,**kwargs)



def naive_sum_test(Ws,X,**kwargs):
	
	NN_=lambda X:NN_nd(Ws,X,**kwargs)
	
	n,d=X.shape[-2:]
	I=jnp.eye(n)

	out=0
	for p in itertools.permutations(I):
		P=jnp.array(p)
		sign=jnp.linalg.det(P)
		
		out=out+sign*NN_(jnp.dot(P,X))			

	return out






acs={'tanh','ReLU','DReLU'}

def test_multilayer(d=3,n=5,layers=5,samples=100,checkagainstnaive=False):	
	m=n*d
	key=jax.random.PRNGKey(0)
	key1,key2,key3,key4,*keys=jax.random.split(key,1000)
	
	W=jax.random.normal(key1,(m,n,d))*jnp.sqrt(2/m)
	Ws=[jax.random.normal(keys[i],(m,m))*jnp.sqrt(2/m) for i in range(layers-2)]
	w=jax.random.normal(key2,(1,m))*jnp.sqrt(2/m)
	Ws=[W]+Ws+[w]
	
	X=jax.random.normal(key3,(samples,n,d))


	antisymmetrized={ac:GPU_sum.sum_perms_multilayer(Ws,X,ac)/jnp.sqrt(math.factorial(n)) for ac in acs}
	nonsymmetrized={ac:NN_nd(Ws,X,ac=ac) for ac in acs}

	#if checkagainstnaive==True:
	#	print(naive_sum_test(Ws,X))
	#	print(naive_sum_test(Ws,X,ac='tanh'))
	
	return antisymmetrized,nonsymmetrized
