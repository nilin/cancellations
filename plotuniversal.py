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
import matplotlib.pyplot as plt



def colorformat(v):
	v=v/jnp.sqrt(jnp.var(v))
	#v=jnp.tanh(v)
	plus=(jnp.sign(v)+1)/2
	colors=jnp.stack([1-plus,0*v,plus],axis=1)
	magnitudes=jnp.abs(v)

	return colors,magnitudes


def plot2(X,Y,ax):

	x=X[:,0,0]
	y=X[:,1,0]

	colors,magnitudes=colorformat(Y)
	ax.scatter(x,y,c=colors,s=magnitudes)	


def plot3(X,Y,ax):

	x=X[:,0,0]
	y=X[:,1,0]
	z=X[:,2,0]

	colors,magnitudes=colorformat(Y)
	ax.scatter(x,y,z,c=colors,s=magnitudes)



if __name__=='__main__':
	d=1
	n=2

	X=bk.get('data/X_train_n=3_d=1')
	Y=bk.get('data/Y_train_n=3_d=1_m=10')

	samples=int(sys.argv[1])

	fig=plt.gcf()
	ax=fig.add_subplot(projection='3d')
	plot(X,Y,ax,samples)
	plt.show()
