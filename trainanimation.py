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
import plotuniversal


def plottarget(ax,plotfn,n,samples=10000,AS=True):
	d=1
	X=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
	#fnY='data/Y_test_n='+str(n)+'_d='+str(d)+'_m=1' if AS else 'data/nsY_test_n='+str(n)+'_d='+str(d)+'_m=1'
	fnY='data/Y_test_n='+str(n)+'_d='+str(d)+'_m=1'
	Y=bk.get(fnY)
	X=X[:samples]
	Y=Y[:samples]
	plotfn(X,Y,ax)	


def plottrained(ax,plotfn,Wb,samples=10000,AS=True):
	W,b=Wb
	n,d=W[0].shape[-2:]
	X=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
	X=X[:samples]
	Y=universality.sumperms(W,b,X) if AS else universality.nonsym(W,b,X)
	plotfn(X,Y,ax)	

def animate(n,AS=True):
	plt.ion()
	fig=plt.figure()

	plotfn=plotuniversal.plot2 if n==2 else plotuniversal.plot3

	if n==2:
		ax1=fig.add_subplot(1,2,1)
		ax2=fig.add_subplot(1,2,2)
	if n==3:
		ax1=fig.add_subplot(1,2,1,projection='3d')
		ax2=fig.add_subplot(1,2,2,projection='3d')


	plottarget(ax1,plotfn,n,AS=AS,samples=10000)

	m=10
	variables={'d':1,'n':n,'m':m}	
	hist=bk.get('data/hists/'+bk.formatvars_(variables))

	while True:
		ax2.cla()
		plt.pause(0.1)
		for Wb in hist:
			ax2.cla()
			plottrained(ax2,plotfn,Wb,AS=AS,samples=10000)
			plt.pause(0.1)
	

def saveanimation(n):
	m=10
	variables={'d':1,'n':n,'m':m}	
	hist=bk.get('data/hists/'+bk.formatvars_(variables))

	for i,Wb in enumerate(hist):

		print(str(i)+' of '+str(len(hist)))

		fig=plt.figure()

		plotfn=plotuniversal.plot2 if n==2 else plotuniversal.plot3

		if n==2:
			ax1=fig.add_subplot(1,2,1)
			ax2=fig.add_subplot(1,2,2)
		if n==3:
			ax1=fig.add_subplot(1,2,1,projection='3d')
			ax2=fig.add_subplot(1,2,2,projection='3d')


		plottarget(ax1,plotfn,n,10000)
		plottrained(ax2,plotfn,Wb,10000)

		m=10
		
		plt.savefig('animation/n='+str(n)+'/'+str(i))
		

if __name__=="__main__":
	n=int(sys.argv[1])
	#saveanimation(n)
	animate(n)
	#animate(n,AS=False)
