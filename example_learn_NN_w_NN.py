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
import pdb
import math
import dashboard as db
import time
import targetfunctions
import AS_tools



bgvars=set(globals().keys())

n=5
d=1
samples=10000
testsamples=1000
widths=100
targetwidths=[25,25,25]
initfromfile=None
plotfineness=1000
period=5

bk.getparams(globals(),sys.argv)


fgvars=set(globals().keys())-bgvars-{'bgvars'}







"""
press Ctrl-C to stop training
"""
def initandtrain(X,Y,widths,dashboard=bk.emptydashboard,**kwargs): 
	T=learning.TrainerWithValidation(widths,X,Y,fractionforvalidation=.1,**kwargs)
	T.tracker.add_listener(dashboard)
	db.clear()

	stopwatch=bk.Stopwatch()
	try:
		while True:
			if stopwatch.elapsed()<period:
				try:
					T.step()
					dashboard.refresh('stopwatch',stopwatch.elapsed())
				except KeyboardInterrupt:
					on_pause(T)
			else:
				stopwatch.tick()
				do_periodic(T)

				
	except KeyboardInterrupt:
		print('\nEnding.\n')




def do_periodic(trainer):
	trainer.checkpoint()
	try:
		saveplots(trainer)
	except Exception as e:
		bk.log(str(e))


def on_pause(trainer):

	msg='\nPaused. Press (Enter) to continue training.'
	msg=msg+'\nEnter (set) to set training variable.'	
	msg=msg+'\nEnter (p) to show plots.'	
	msg=msg+'\nEnter (q) to end training.\n'

	while True:
		
		db.clear()
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
		
def saveplots(trainer):

	plt.close('all')

	test=bk.get('data/hist')

	learnedAS=learning.AS_from_hist('data/hist')
	X_test=rnd.uniform(k1,(testsamples,n,d),minval=-1,maxval=1)

	fig1,ax1=plt.subplots(1)
	fig2,(ax21,ax22)=plt.subplots(1,2)

	pt.plotalongline(ax1,targetAS,learnedAS,X_test,fineness=plotfineness)
	pt.ploterrorhist(ax21,'data/hist')
	pt.ploterrorhist(ax22,'data/hist',logscale=True)

	figpath='plots/started '+trainer.tracker.ID+' | '+bk.formatvars(vardefs,ignore={'plotfineness','minibatchsize','initfromfile','testsamples','d','samples'})+'/'
	bk.savefig(figpath+'plot.pdf',fig1)
	bk.savefig(figpath+'losses.pdf',fig2)
	bk.savefig('plots/plot.pdf',fig1)
	bk.savefig('plots/losses.pdf',fig2)
	
	return fig1,fig2,figpath







vardefs={k:globals()[k] for k in fgvars}



k0,k1,k2,*keys=rnd.split(rnd.PRNGKey(0),100)
X_train=rnd.uniform(k0,(samples,n,d),minval=-1,maxval=1)



targetAS=targetfunctions.gen_fixed_AS_NN(n,d,targetwidths)
targetAS=util.normalize(targetAS,X_train[:100])


Y_train=targetAS(X_train)



dashboard=db.Dashboard()
dashboard.addtext(*bk.formatvars(vardefs,'\n').split('\n'))
dashboard.addspace()
dashboard.addtext('time to next validation set/save')
dashboard.addbar(lambda defs,hists:1-defs['stopwatch']/period,style=db.dash)
dashboard.addspace(1)
dashboard.addtext(lambda defs,hists:'{:,} samples left in epoch'.format(defs['minibatches left']*defs['minibatchsize']))
dashboard.addbar(lambda defs,hists:defs['minibatches left']/defs['minibatches'],style=db.dash)
dashboard.addspace(5)
dashboard.addtext('training loss of last minibatch, 10, 100 minibatches')
dashboard.addbar(lambda defs,hists:defs['minibatch loss'])
dashboard.addbar(lambda defs,hists:np.average(np.array(hists['minibatch loss'])[-10:]),style=db.box)
dashboard.addbar(lambda defs,hists:np.average(np.array(hists['minibatch loss'])[-100:]),style=db.box)
dashboard.addspace()
dashboard.addtext('validation loss')
dashboard.addbar(lambda defs,hists:defs['validation loss'])

bk.bgtracker.add_listener(dashboard)	

initandtrain(X_train,Y_train,widths,dashboard=dashboard,initfromfile=initfromfile)


