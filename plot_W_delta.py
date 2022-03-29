import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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






def plot_sphere(c,r,ax):
	theta=jnp.arange(0,2*math.pi,.1)
	phi=jnp.arange(0,math.pi/2,0.1)
	X=jnp.cos(theta[:,None])*jnp.cos(phi[None,:])
	Y=jnp.sin(theta[:,None])*jnp.cos(phi[None,:])
	Z=jnp.sin(phi[None,:])
	ax.plot_surface(r*X+c[0],r*Y+c[1],r*Z+c[2],color='b')
	ax.plot_surface(r*X+c[0],r*Y+c[1],-r*Z+c[2],color='b')


def plot_spheres(centers,radii):
	fig=plt.figure()
	ax=fig.add_subplot(projection='3d')

	n=radii.size
	for i in range(n):
		c,r=centers[i],radii[i]
		plot_sphere(c,r,ax)

	plt.show()

def plot_packing(centers):
	radii=.5*util.mindist_per_i(centers)
	plot_spheres(centers,radii)
	


if __name__=='__main__':

	data=bk.getdata('w_packing')
	w=data['ws'][int(input('input n<20 '))]
	plot_packing(w)


"""
data=bk.getdata('W_separated, seed=0')
instances,d,n_,deltas=data['instances'],data['d'],data['n_'],data['deltas']
plt.plot(n_,[deltas[n] for n in n_],'r')
print(deltas)

seed=0
key=jax.random.PRNGKey(seed)
key0,*keys=jax.random.split(key,1000)
instances=100

Ws={n:jax.random.normal(keys[n],(instances,n,d))/jnp.sqrt(n*d) for n in n_}
deltas={n:util.L2norm(util.mindist(Ws[n])) for n in n_}
plt.plot(n_,[deltas[n] for n in n_],'b')
print(deltas)


C=1
plt.plot(n_,[C/n for n in n_])


plt.yscale('log')
plt.show()
"""
