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
import numpy as np




				
		


def testerror(Wb,samples=100):
	W,b=Wb
	m,n,d=W[0].shape
	X=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
	Y=bk.get('data/Y_test_n='+str(n)+'_d='+str(d)+'_m=1')

	X=X[:samples]
	Y=Y[:samples]

	return universality.batchloss(Wb,X,Y)/universality.lossfn(Y,0)
	
def testerrorNS(Wb,samples=100):
	W,b=Wb
	m,n,d=W[0].shape
	X=bk.get('data/X_test_n='+str(n)+'_d='+str(d))
	Z=bk.get('data/Z_test_n='+str(n)+'_d='+str(d)+'_m=1')

	X=X[:samples]
	Z=Z[:samples]

	return universality.batchlossNS(Wb,X,Z)/universality.lossfn(Z,0)



def ploterrorhist(n,ax,fn,plotmode):
	hist=bk.get(fn)
	errorhist=[testerror(Wb) if plotmode=='AS' else testerrorNS(Wb) for Wb in hist]
	error=errorhist[-1]

	if error<.5:
		ax.plot(errorhist,label=str(n))
	else:
		ax.plot(errorhist,ls='dotted',label=str(n))

	




if __name__=="__main__":

	trainmode=input('training mode: ')
	outfn=input('output file name: ')

	fig,axs=plt.subplots(nrows=2,ncols=2)

	print(axs)

	for d,axrow in zip([1,3],[axs[0],axs[1]]):

		for plotmode,ax in zip(['AS','NS'],axrow):

			ax.set_ylim((0,2))	

			print('d='+str(d))
			for n in range(1,8):
				print('n='+str(n))
				fn='data/hists/'+trainmode+'_'+bk.formatvars_({'d':d,'n':n,'m':100})
				ploterrorhist(n,ax,fn,plotmode)
			ax.legend()
	
	bk.savefig('univplots/'+outfn+'.pdf')
		

	
	





