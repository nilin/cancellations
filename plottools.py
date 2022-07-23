import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
import optax
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import learning




def linethrough(x,fineness=1000):
	corner=np.zeros_like(x)
	corner[0][0]=1

	x_rest=(1-corner)*x
	I=jnp.arange(-1,1,2/fineness)

	X=I[:,None,None]*corner[None,:,:]+x_rest[None,:,:]
	return I,X



				
def plotalongline(targetAS,learnedAS,X,**kwargs):

	Y=targetAS(X)
	x0=X[jnp.argmax(Y**2)]
	I,x=linethrough(x0,**kwargs)

	fig,ax=plt.subplots(1)
	ax.plot(I,targetAS(x),'b',label='target')
	ax.plot(I,learnedAS(x),'r',label='learned')
	ax.legend()
	return fig



def ploterrorhist(path):
	timestamps,trainerrors,valerrors=[learning.losses_from_hist(path)[k] for k in ['timestamps','trainerrorhist','valerrorhist']]
	mins=timestamps/60
	fig,ax=plt.subplots(1)
	ax.plot(mins[1:]/2+mins[:-1]/2,trainerrors[1:],'rd:',label='training error')
	ax.plot(mins,valerrors,'bo-',label='validation error')
	ax.legend()
	ax.set_xlabel('minutes')
	ax.set_yscale('log')
	return fig





	

#
#def animate(n,AS=True):
#	plt.ion()
#	fig=plt.figure()
#
#	plotfn=plotuniversal.plot2 if n==2 else plotuniversal.plot3
#
#	if n==2:
#		ax1=fig.add_subplot(1,2,1)
#		ax2=fig.add_subplot(1,2,2)
#	if n==3:
#		ax1=fig.add_subplot(1,2,1,projection='3d')
#		ax2=fig.add_subplot(1,2,2,projection='3d')
#
#
#	plottarget(ax1,plotfn,n,AS=AS,samples=10000)
#
#	m=10
#	variables={'d':1,'n':n,'m':m}	
#	hist=bk.get('data/hists/'+bk.formatvars_(variables))
#
#	while True:
#		ax2.cla()
#		plt.pause(0.1)
#		for Wb in hist:
#			ax2.cla()
#			plottrained(ax2,plotfn,Wb,AS=AS,samples=10000)
#			plt.pause(0.1)
#	
#

if __name__=='__main__':
	ploterrorhist('data/hist');plt.show()
