import numpy as np
import math
import itertools
import pickle
import time
import copy
import util
import jax
import jax.numpy as jnp
import permutations
import optax
	

Wtypes={'s':'separated','n':'normal','ss':'separated small','ns':'normal small'}

apply_tau_=lambda W,X,activation=util.ReLU:activation(jnp.matmul(util.flatten_nd(W),util.flatten_nd(X).T))
apply_tau=apply_tau_

def w_to_alpha(W,activation):
	F=lambda X:apply_tau_(W,X,activation)
	return antisymmetrize(F)

def apply_alpha(W,X,activation=util.ReLU):
	alpha_w=w_to_alpha(W,activation)
	return alpha_w(X)

def antisymmetrize(f):
	def antisymmetric(X):
		y=jnp.zeros(f(X).shape)
		n=X.shape[-2]
		for P in itertools.permutations(jnp.identity(n)):
			sign=jnp.linalg.det(P)
			PX=jnp.swapaxes(jnp.dot(jnp.array(P),X),0,-2)
			y+=sign*f(PX)
		return y/jnp.sqrt(math.factorial(n))
	return antisymmetric


