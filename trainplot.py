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
	

def ploterrorhist(variables,ax):
	hist=bk.get('data/hists/'+bk.formatvars_(variables))
	errorhist=[testerror(Wb) for Wb in hist]
	error=errorhist[-1]

	n=variables['n']

	if error<.5:
		ax.plot(errorhist,label=str(n))
	else:
		ax.plot(errorhist,ls='dotted',label=str(n))

	




if __name__=="__main__":

	#variables=bk.formatvars(sys.argv[1:])
	#ploterrorhist(variables)

	fn=input('file name: ')

	fig,axs=plt.subplots(1,2)

	for d,ax in zip([1,3],[axs[0],axs[1]]):

	#d=sys.argv[1]
		ax.set_ylim((0,2))	

		print('d='+str(d))
		for n in range(1,8):
			print('n='+str(n))
			ploterrorhist({'d':d,'n':n,'m':10},ax)

		ax.legend()


	plt.savefig('animation/'+fn+'.pdf')
		

	
	





