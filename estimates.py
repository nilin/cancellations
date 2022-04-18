
import math
import pickle
import jax
import jax.numpy as jnp
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat


i_=1

F_HS=lambda t: 1/(i_*2*math.pi*t)
F_ReLU=lambda t: -1/(4*math.pi**2*jnp.square(t))
F_tanh=lambda t: math.pi/(i_*jnp.sinh(math.pi**2*t))

gausskernel=lambda t: jnp.exp(-2*math.pi**2*jnp.square(t))



def getF(ac_name):
	return globals()['F_'+ac_name]




def evaluationpointsandweights(thetas,density):
	step=.25
	morethetas=jnp.arange(jnp.max(thetas)+step,10000,step)

	thetas_=jnp.concatenate([thetas,morethetas])
	dthetas=thetas_[1:]-thetas_[:-1]
	density_=jnp.concatenate([density,jnp.ones((morethetas.size,))])
	weights=jnp.multiply(dthetas,density_[1:])

	return thetas_[1:],weights
	


"""
for n,density in densities.items():

for n,x in dets.items():
	print(n)
	print(x)
"""




def fourierestimate(ac_name,n,d):
	oscdata=loadmat('data/oscdets d='+str(d)+'.mat')
	nrange=np.squeeze(oscdata['nrange']).astype(int)
	thetas=jnp.squeeze(jnp.array(oscdata['thetas']))/(2*math.pi)
	densities={nrange[k]:jnp.array(oscdata['dets'][k]) for k in range(nrange.size)}
	
	thetas_,weights=evaluationpointsandweights(thetas,densities[n])

	f_hat=getF(an_name)
	return jnp.inner(jnp.square(f_hat(thetas_)),weights)

