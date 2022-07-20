import jax.numpy as jnp
import jax
import math
import itertools
import util
import legacy.permutations as lp
import numpy as np
import pdb

def assertequal(y,z,blockdim=0):
	print('comparing')
	print(jnp.stack([y,z],axis=-blockdim-1))
	print('yes, they agree')
	return util.relloss(y,z)<.001



def naiveAS(NS,X):
	samples,n,d=X.shape
	p0=list(range(n))
	out=0
	for p in itertools.permutations(p0):
		sign=permutations.sign(p)
		PX=X[:,p,:]
		out=out+sign*NS(X)
	return out


def verify_antisymmetrization(AS,NS,X):
	Y=AS(X)
	Z=naiveAS(NS,X)
	assertequal(Y,Z)


def verify_antisymmetric(AS,X):
	n=X.shape[-2]
	Y=AS(X)
	for _ in range(25):
		p=np.random.permutation(n)
		sign=lp.sign(p)

		PX=np.array(X)[:,p,:]
		assertequal(AS(PX),Y*sign)

	

def testperms(Ps,signs):
	n=Ps.shape[-1]
	_Ps_,_signs_=lp.gen_complementary_Perm_seqs([n])[0]
	assertequal(Ps,_Ps_,2)
	assertequal(signs,_signs_)

def testpermtuples(ps,signs):
	n=ps.shape[-1]
	_ps_,_signs_=lp.gen_complementary_perm_seqs([n])[0]
	assertequal(ps,_ps_,1)
	assertequal(signs,_signs_)







#
#
#
#
#def NN_(Ws,bs,X,ac):
#	activation=util.activations[ac]
#	X=X.T
#	for W,b in zip(Ws[:-1],bs):
#		X=activation(jnp.dot(W,X)+b)
#	return jnp.dot(Ws[-1],X)
#
#def NN(Ws,X,ac):
#	bs=[jnp.zeros((W.shape[0],)) for W in Ws[:-1]]
#	return NN_(Ws,bs,X,ac)
#
##def get_NN_nd(ac):
##	@jax.jit
##	def NN_nd(Ws,X):
##		n,d=X.shape[-2:]
##		flatW=jnp.reshape(Ws[0],Ws[0].shape[:-2]+(n*d,))
##		flatX=jnp.reshape(X,X.shape[:-2]+(n*d,))
##		Ws_=[flatW]+Ws[1:]
##
##		return NN(Ws_,flatX,ac)
##	return NN_nd
#
#def get_NN_nd(ac):
#	@jax.jit
#	def NN_nd(Ws,bs,X):
#		n,d=X.shape[-2:]
#		flatW=jnp.reshape(Ws[0],Ws[0].shape[:-2]+(n*d,))
#		flatb=jnp.reshape(bs[0],bs[0].shape[:-2]+(n*d,))
#		flatX=jnp.reshape(X,X.shape[:-2]+(n*d,))
#		Ws_=[flatW]+Ws[1:]
#		bs_=[flatb]+bs[1:]
#
#		return NN_(Ws_,bs_,flatX,ac)
#	return NN_nd
#
#
#
#acs={'tanh','DReLU'}
#
#def test_multilayer(d=3,n=5,layers=5,samples=100,checkagainstnaive=False):	
#	m=n*d
#	key=jax.random.PRNGKey(0)
#	key1,key2,key3,key4,*keys=jax.random.split(key,1000)
#	
#	W=jax.random.normal(key1,(m,n,d))*jnp.sqrt(2/m)
#	Ws=[jax.random.normal(keys[i],(m,m))*jnp.sqrt(2/m) for i in range(layers-2)]
#	w=jax.random.normal(key2,(1,m))*jnp.sqrt(2/m)
#	Ws=[W]+Ws+[w]
#	
#	X=jax.random.normal(key3,(samples,n,d))
#
#
#	antisymmetrized={ac:GPU_sum.sum_perms_multilayer(Ws,X,ac)/jnp.sqrt(math.factorial(n)) for ac in acs}
#	nonsymmetrized={ac:NN_nd(Ws,X,ac=ac) for ac in acs}
#
#	#if checkagainstnaive==True:
#	#	print(antisymmetrized['tanh'])
#	#	print(naive_sum_test(Ws,X,ac='tanh'))
#	
#	return antisymmetrized,nonsymmetrized
#
#
#
#def naive_sum_test(Ws,X,**kwargs):
#	
#	NN_=lambda X:NN_nd(Ws,X,**kwargs)
#	
#	n,d=X.shape[-2:]
#	I=jnp.eye(n)
#
#	out=0
#	for p in itertools.permutations(I):
#		P=jnp.array(p)
#		sign=jnp.linalg.det(P)
#		
#		out=out+sign*NN_(jnp.dot(P,X))			
#
#	return out
#
