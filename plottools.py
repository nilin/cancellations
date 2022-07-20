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





def linethrough(x):
	corner=np.zeros_like(x)
	corner[0][0]=1

	x_rest=(1-corner)*x
	I=jnp.arange(-1,1,.01)

	X=I[:,None,None]*corner[None,:,:]+x_rest[None,:,:]
	return I,X
				
	

def testerror(Ws,bs,X,Y):


	return train.batchloss(Wb,X,Y)/train.lossfn(Y,0)
	





def ploterrorhist(n,ax,fn,plotmode):
	hist=bk.get(fn)
	errorhist=[testerror(Wb) if plotmode=='AS' else testerrorNS(Wb) for Wb in hist]
	error=errorhist[-1]

	if error<.5:
		ax.plot(errorhist,label=str(n))
	else:
		ax.plot(errorhist,ls='dotted',label=str(n))

	

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
	

#
#if __name__=="__main__":
#
#	nmax=int(sys.argv[1])
#
#	trainmode=input('training mode: ')
#	outfn=input('output file name: ')
#
#	fig,axs=plt.subplots(nrows=2,ncols=2)
#
#	print(axs)
#
#	for d,axrow in zip([1,3],[axs[0],axs[1]]):
#
#		for plotmode,ax in zip(['AS','NS'],axrow):
#
#			ax.set_ylim((0,2))	
#
#			print('d='+str(d))
#			for n in range(1,nmax+1):
#				print('n='+str(n))
#				fn='data/hists/'+trainmode+'_'+bk.formatvars_({'d':d,'n':n,'m':100})
#				ploterrorhist(n,ax,fn,plotmode)
#			ax.legend()
#	
#	bk.savefig('univplots/'+outfn+'.pdf')
#		

	
	





