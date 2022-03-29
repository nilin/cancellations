import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import pickle
import time
import bookkeep
import copy
import jax
import jax.numpy as jnp
import optax
import util
import bookkeep as bk


seed=0
key=jax.random.PRNGKey(seed)
key0,*keys=jax.random.split(key,1000)

data=bk.getdata('W_separated')
instances,d,n_,deltas=data['instances'],data['d'],data['n_'],data['deltas']
plt.plot(n_,[deltas[n] for n in n_],'r')
print(deltas)

Ws={n:jax.random.normal(keys[n],(instances,n,d))/jnp.sqrt(n*d) for n in n_}
deltas={n:util.L2norm(util.mindist(Ws[n])) for n in n_}
plt.plot(n_,[deltas[n] for n in n_],'b')
print(deltas)


plt.yscale('log')
plt.show()
