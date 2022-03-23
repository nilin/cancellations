import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
import seaborn as sns
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import optax
import util
import sys
import os
import shutil
import cancellation as canc
import proxies
import parallelpartialsum as p_sums
import permutations




def detexp(W,X):
	n=W.shape[-2]
	print(n)
	dots=jnp.swapaxes(jnp.inner(W,X),1,2)
	return jnp.linalg.det(jnp.exp(dots))/jnp.sqrt(math.factorial(n))


"""
combine_partial_sums.py ReLU n 10
"""


Wtype=util.Wtypes[input('W type: (n)normal, (s)eparated, ns=(n)ormall (s)mall, (s)eparated (s)mall :')]
nmin,nmax=[int(inp) for inp in input('range n_min n_max inclusive :').split()]

Ws,Xs=bk.getdata(Wtype+'/WX')['Ws'],bk.getdata(Wtype+'/WX')['Xs']

n_=list(range(nmin,nmax+1))
sums=[detexp(Ws[n],Xs[n]) for n in n_]

bk.savedata({'range':n_,'data':sums,'norms':[util.L2norm(Y) for Y in sums]},Wtype+'/exp')
