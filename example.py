# nilin


import config as cfg
import sys
import jax
import jax.numpy as jnp
import jax.random as rnd
import AS_functions
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
import AS_tools







params={
'learnertype':'AS_NN',
'targettype':'SlaterSumNN',
'n':6,
'd':1,
'samples_train':10000,
'samples_val':100,
'samples_test':1000,
'learnerwidths':100,
'targetwidths':[25,25,25],
'plotfineness':1000,
'checkpoint_interval':5
}


redefs=cfg.get_cmdln_args()
globals().update(params)
globals().update(redefs)
fgvars=cfg.orderedunion(params,redefs)




def initandtrain(learner,X,Y,dashboard=cfg.emptydashboard,**kwargs): 

	T=learning.TrainerWithValidation(learner,X,Y,validationbatchsize=samples_val,**kwargs)
	T.tracker.add_listener(dashboard)
	db.clear()

	stopwatch=cfg.Stopwatch()
	while True:
		if stopwatch.elapsed()<checkpoint_interval:
			T.step()
			dashboard.refresh('stopwatch',stopwatch.elapsed())
		else:
			stopwatch.tick()
			do_periodic(T)



def do_periodic(trainer):
	trainer.checkpoint()
	try:
		saveplots(trainer)
	except Exception as e:
		cfg.log(str(e))


def saveplots(trainer):
	plt.close('all')
	test=cfg.get('data/hist')

	learnedAS=learning.AS_from_hist('data/hist')
	X_test=rnd.uniform(k1,(testsamples,n,d),minval=-1,maxval=1)

	fig1,ax1=plt.subplots(1)
	fig2,(ax21,ax22)=plt.subplots(1,2)

	pt.plotalongline(ax1,targetAS,learnedAS,X_test,fineness=plotfineness)
	pt.ploterrorhist(ax21,'data/hist')
	pt.ploterrorhist(ax22,'data/hist',logscale=True)

	figpath='plots/started '+trainer.tracker.ID+' | '+cfg.formatvars(vardefs,ignore={'plotfineness','minibatchsize','initfromfile','testsamples','d','samples'})+'/'
	cfg.savefig(figpath+'plot.pdf',fig1)
	cfg.savefig(figpath+'losses.pdf',fig2)
	cfg.savefig('plots/plot.pdf',fig1)
	cfg.savefig('plots/losses.pdf',fig2)
	
	return fig1,fig2,figpath







vardefs={k:globals()[k] for k in fgvars}


k0,k1,k2,*keys=rnd.split(rnd.PRNGKey(0),100)
X_train=rnd.uniform(k0,(samples_train+samples_val,n,d),minval=-1,maxval=1)



targetAS=AS_functions.gen_static_AS_NN(n,d,targetwidths)
targetAS=util.normalize(targetAS,X_train[:100])


Y_train=targetAS(X_train)



D=db.Dashboard()
D.addtext(*[name+'='+str(globals()[name]) for name in fgvars])
D.addspace()
D.addtext('time to next validation set/save')
D.addbar(lambda defs,hists:1-defs['stopwatch']/checkpoint_interval,style=db.dash)
D.addspace(1)
D.addtext(lambda defs,hists:'{:,} samples left in epoch'.format(defs['minibatches left']*defs['minibatchsize']))
D.addbar(lambda defs,hists:defs['minibatches left']/defs['minibatches'],style=db.dash)
D.addspace(5)
D.addtext('training loss of last minibatch, 10, 100 minibatches')
D.addbar(lambda defs,hists:defs['minibatch loss'])
D.addbar(lambda defs,hists:np.average(np.array(hists['minibatch loss'])[-10:]),style=db.box)
D.addbar(lambda defs,hists:np.average(np.array(hists['minibatch loss'])[-100:]),style=db.box)
D.addspace()
D.addtext('validation loss')
D.addbar(lambda defs,hists:defs['validation loss'])

cfg.bgtracker.add_listener(D)	

learner=AS_functions.init_AS_NN(n,d,learnerwidths)
initandtrain(learner,X_train,Y_train,dashboard=D)


