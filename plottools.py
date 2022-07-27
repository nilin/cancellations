import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
import optax
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import learning
import pdb
from collections import deque



def linethrough(x,fineness=1000):
	corner=np.zeros_like(x)
	corner[0][0]=1

	x_rest=(1-corner)*x
	I=jnp.arange(-1,1,2/fineness)

	X=I[:,None,None]*corner[None,:,:]+x_rest[None,:,:]
	return I,X



				
def plotalongline(ax,target,learned,X,**kwargs):
	Y=target(X)
	x0=X[jnp.argmax(Y**2)]
	I,x=linethrough(x0,**kwargs)

	ax.plot(I,target(x),'b',label='target')
	ax.plot(I,learned(x),'r',label='learned')
	ax.legend()



def partition(bins,x,*ys):
	bin_nrs=np.digitize(x,bins)
	blocks=[np.where(bin_nrs==b)[0] for b in range(bin_nrs[-1]+1)]
	return [[np.array(y)[I] for I in blocks] for y in (x,)+ys]

	

def ploterrorhist(ax,hists,logscale=False):

	train=hists['minibatch loss']
	test=hists['test loss']
	t_train_blocks,train_loss_blocks=partition(test['timestamps'],train['timestamps'],train['vals'])
	t_train,train_loss=[np.average(t) for t in t_train_blocks],[np.average(l) for l in train_loss_blocks]
	ax.plot(t_train,train_loss,'rd--',label='training loss')
	ax.plot(test['timestamps'],test['vals'],'bo-',label='test loss')
	
	ax.legend()
	ax.set_xlabel('seconds')
	#ax.set_xlabel('minutes')
	if logscale:
		ax.set_yscale('log')
	else:
		ax.set_ylim(0,1)





	

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
#	hist=cfg.get('data/hists/'+cfg.formatvars_(variables))
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
	fig,ax=plt.subplots(1)
	ploterrorhist(ax,'data/hist');plt.show()
	#print(partition([10,20],np.arange(100),np.arange(0,1000,10)))
