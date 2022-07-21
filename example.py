# nilin


import bookkeep as bk
import sys
import jax
import jax.numpy as jnp
import jax.random as rnd
import targetfunctions as targets
import learning
import plottools as pt
import mcmc
import matplotlib.pyplot as plt






if __name__=='__main__':

	requiredvars={'n','samples','minibatchsize','widths'}

	n=5
	d=1
	samples=10000
	testsamples=1000
	minibatchsize=100
	widths=[50,100]
	trainmode='AS'
	batchmode='minibatch'


	bk.getparams(globals(),sys.argv,requiredvars)
	print(2*'\n'+'\n'.join([name+' = '+str(globals()[name]) for name in requiredvars])+2*'\n')



	targetAS=targets.HermiteSlater(n,'H',1/8)

	k0,k1,k2,*keys=rnd.split(rnd.PRNGKey(0),100)
	X_train=rnd.uniform(k0,(samples,n,d),minval=-1,maxval=1)
	Y_train=targetAS(X_train)

	bk.save([X_train,Y_train],'data/XY')


	
	def on_pause(trainer):
		trainer.savehist('data/hist')
		learnedAS=learning.AS_from_hist('data/hist')
		X_test=rnd.uniform(k1,(testsamples,n,d),minval=-1,maxval=1)
		fig1=pt.plotalongline(targetAS,learnedAS,X_test)
		fig2=pt.ploterrorhist('data/hist')
		bk.savefig('plots/alongline.pdf',fig1)
		bk.savefig('plots/trainingerror.pdf',fig2)

		msg='\nPaused. Press (Enter) to continue training.'
		msg=msg+'\nEnter (p) to show plots.'	
		msg=msg+'\nEnter (b/m) to toggle batch/minibatch descent.'	
		msg=msg+'\nEnter (q) to end training.\n'

		while True:
			inp=input(msg)
			if inp=='': break
			if inp=='p':
				print('\nShowing plots in background\n')
				plt.show()
			if inp in {'b','m'}: trainer.set_batchmode({'b':'batch','m':'minibatch'}[inp])
			if inp in {'a','n'}: trainer.set_symmode({'a':'AS','n':'NS'}[inp])
			if inp=='q': raise KeyboardInterrupt



	learning.initandtrain('data/XY','data/hist',widths,minibatchsize,batchmode=batchmode,action_on_pause=on_pause)


