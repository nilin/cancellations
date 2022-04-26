import jax.numpy as jnp
import math
import itertools
import util




def NN(Ws,X,ac='ReLU'):
	activation={'ReLU':util.ReLU,'tanh':jnp.tanh,'HS':util.heaviside}[ac]
	X=X.T
	for W in Ws[:-1]:
		X=activation(jnp.dot(W,X))
	return jnp.dot(Ws[-1],X)


def NN_nd(Ws,X,**kwargs):
	n,d=X.shape[-2:]
	flatW=jnp.reshape(Ws[0],Ws[0].shape[:-2]+(n*d,))
	flatX=jnp.reshape(X,X.shape[:-2]+(n*d,))
	Ws_=[flatW]+Ws[1:]

#	print(Ws)
#	print(Ws_)
#
#	print(X)
#	print(flatX)

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


