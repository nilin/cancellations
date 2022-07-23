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
import numpy as np
import util

import screendraw







"""
press Ctrl-C to stop training
"""
def initandtrain(data_in_path,widths,action_each_epoch=util.donothing,action_on_pause=util.donothing,**kwargs): 
	X,Y=bk.get(data_in_path)
	T=learning.TrainerWithValidation(widths,X,Y,fractionforvalidation=.01,microbatchsize=5)
	screendraw.clear()
	try:
		while True:
			try:
				print('\nEpoch '+str(T.epochsdone()))
				T.epoch()
				action_each_epoch(T)
			except KeyboardInterrupt:
				action_on_pause(T)
				continue
	except KeyboardInterrupt:
		print('\nEnding.\n')


if __name__=='__main__':


	bgvars=set(globals().keys())

	n=8
	d=1
	samples=100000
	testsamples=1000
	widths=[25,25,25]
	initfromfile=None
	plotfineness=1000

	bk.getparams(globals(),sys.argv)


	fgvars=set(globals().keys())-bgvars-{'bgvars'}

	vardefs={k:globals()[k] for k in fgvars}
	print(bk.formatvars(vardefs,'\n'))	



	k0,k1,k2,*keys=rnd.split(rnd.PRNGKey(0),100)
	X_train=rnd.uniform(k0,(samples,n,d),minval=-1,maxval=1)

	
	
	targetAS=targets.HermiteSlater(n,'H',1/8)
	targetAS=util.normalize(targetAS,X_train[:100])


	Y_train=targetAS(X_train)
	bk.save([X_train,Y_train],'data/XY')

	
	bk.addtext('training loss of last minibatch, 10, 100 minibatches, epoch up to now')
	bk.addbar(lambda defs:np.average(np.array(defs['minibatch losses'])[-1:]))
	bk.addbar(lambda defs:np.average(np.array(defs['minibatch losses'])[-10:]))
	bk.addbar(lambda defs:np.average(np.array(defs['minibatch losses'])[-100:]))
	bk.addbar(lambda defs:jnp.average(jnp.array(defs['minibatch losses'])))
	bk.addspace(1)
	bk.addtext('validation loss')
	bk.addbar(lambda defs:defs['validation loss'])
	bk.addspace(1)
	bk.addtext(lambda defs:'{:,} samples done'.format(defs['samplesdone']))
	bk.addbar(lambda defs:defs['samplesdone']/defs['samples'])
	

	def saveplots(trainer):
	
		plt.close('all')
		trainer.checkpoint()

		learnedAS=learning.AS_from_hist('data/hist')
		X_test=rnd.uniform(k1,(testsamples,n,d),minval=-1,maxval=1)

		fig1=pt.plotalongline(targetAS,learnedAS,X_test,fineness=plotfineness)
		fig2=pt.ploterrorhist('data/hist')
		figpath='plots/started '+trainer.ID+' | '+bk.formatvars(vardefs,ignore={'plotfineness','minibatchsize','initfromfile','testsamples','d','samples'})+'/'
		bk.savefig(figpath+'plot.pdf',fig1)
		bk.savefig(figpath+'losses.pdf',fig2)
		bk.savefig('plots/plot.pdf',fig1)
		bk.savefig('plots/losses.pdf',fig2)
		
		return fig1,fig2,figpath

	
	def on_pause(trainer):

		fig1,fig2,figpath=saveplots(trainer)
		bk.savefig(figpath+str(round(trainer.time_elapsed()))+' s.pdf',fig1)

		msg='\nPaused. Press (Enter) to continue training.'
		msg=msg+'\nEnter (set) to set training variable.'	
		msg=msg+'\nEnter (p) to show plots.'	
		msg=msg+'\nEnter (q) to end training.\n'

		while True:
			inp=input(msg)
			if inp=='': break
			if inp=='p':
				print('\nShowing plots\n')
				fig1.show(); fig2.show()
			if inp=='q': raise KeyboardInterrupt
			if inp=='set':
				name=input('Enter variable name ')
				val=bk.castval(input('Enter value to assign '))
				trainer.setvals(**{name:val})
		screendraw.clear()
		

		

	initandtrain('data/XY',widths,action_each_epoch=saveplots,action_on_pause=on_pause,initfromfile=initfromfile)


