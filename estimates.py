
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
import util
from pathlib import Path
from scipy.io import loadmat


i_=1

F_HS=lambda t: 1/(i_*jnp.sqrt(2*math.pi)*t)
F_ReLU=lambda t: -1/(jnp.sqrt(2*math.pi)*jnp.square(t))
F_tanh=lambda t: i_*jnp.sqrt(math.pi/2)/jnp.sinh((math.pi/2)*t)


# signal processing convention 
#F_HS=lambda t: 1/(i_*2*math.pi*t)
#F_ReLU=lambda t: -1/(4*math.pi**2*jnp.square(t))
#F_tanh=lambda t: math.pi/(i_*jnp.sinh(math.pi**2*t))



def getF(ac_name):
	return globals()['F_'+ac_name]



"""
for n,density in densities.items():

for n,x in dets.items():
	print(n)
	print(x)
"""

def dot(x,y):
	return jnp.nansum(jnp.multiply(x,y))


def fourierestimate(ac_name,n,d):

	data=loadmat('../MATLAB/data/tt d='+str(d)+' n='+str(n)+'.mat')
	thetas=jnp.squeeze(jnp.array(data['thetas']))
	density=jnp.squeeze(jnp.array(data['avgdets']))
	dthetas=thetas[1:]-thetas[:-1];dthetas=jnp.append(dthetas,dthetas[-1])
	weights=dthetas[:,None]*dthetas[None,:]*density

	f_hat=getF(ac_name)
	y=f_hat(thetas)
	integrand=y[:,None]*y[None,:]

	return dot(integrand,weights)/(2*math.pi)

def diagestimate(ac_name,n,d):

	data=loadmat('../MATLAB/data/diag d='+str(d)+' n='+str(n)+'.mat')
	thetas=jnp.squeeze(jnp.array(data['thetas']))
	density=jnp.squeeze(jnp.array(data['avgdets']))
	dthetas=thetas[1:]-thetas[:-1];dthetas=jnp.append(dthetas,dthetas[-1])
	weights=dthetas*density

	f_hat=getF(ac_name)
	integrand=jnp.square(f_hat(thetas))

	return dot(integrand,weights)/jnp.sqrt(2*math.pi)

