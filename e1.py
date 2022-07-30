#
# nilin
# 
# 2022/7
#


import config as cfg
import sys
import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import AS_functions
import learning
import plottools as pt
import mcmc
import matplotlib.pyplot as plt
import numpy as np
import util
from util import ReLU,DReLU,activations
from jax.numpy import tanh
import pdb
import time
import math
import dashboard as db
import time
import testing
import AS_tools
import examplefunctions
import AS_functions as ASf

jax.config.update("jax_enable_x64", True)





cfg.outpaths.add('outputs/e1/{}/'.format(cfg.sessionID))


explanation='Example 1\n'



def run(cmdargs):

	params={
	'targettype':'AS_NN',
	'learnertype':'AS_NN',
	'n':5,
	'd':1,
	'samples_train':10000,
	'samples_test':10000,
	'samples_quicktest':100,
	'targetwidths':[5,25,25,1],
	'learnerwidths':[5,100,1],
	'targetactivation':'tanh',
	'learneractivation':'ReLU',
	'checkpoint_interval':5,
	'timebound':60
	}


	_,redefs=cfg.parse_cmdln_args(cmdargs)
	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)


	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','d','checkpoint_interval'}
	if 'NN' not in targettype: ignore.update({'targetwidths','targetactivation'})

	sessioninfo=explanation+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------
	cfg.log('Generating AS functions.')

	t_args=(n,d,targetwidths,targetactivation) if 'NN' in targettype else (n,)
	target=ASf.init_target(targettype,*t_args) 
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,learneractivation)

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	cfg.log('normalizing target')
	target=util.normalize(target,X[:100])

	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner.as_static(),n,d)

	cfg.log('Generating data Y.')
	Y=target(X)
	Y_test=target(X_test)



	#
	cfg.log('Preparing cross sections for plotting.')
	sections=pt.CrossSections(X,Y,target,3)
	
	plotter=E1plotter()

	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	slate.addtext(lambda *_:'||f|| = {:.2f}'.format(cfg.getval('NS norm')))
	slate.addtext(lambda *_:'||f||/||Af|| = {:.2f}'.format(cfg.getval('norm ratio')))

	
	


	trainer=learning.Trainer(learner,X,Y)
	sc0=cfg.Scheduler(cfg.expsched(.1,timebound))
	sc1=cfg.Scheduler(cfg.periodicsched(1,timebound))
	sc2=cfg.Scheduler(cfg.periodicsched(10,timebound))
	cfg.log('\nStart training.\n')


	while True:
		trainer.step()
		cfg.pokelisteners('refresh')

		if sc0.dispatch():
			trainer.checkpoint()

		if sc1.dispatch():
			cfg.trackcurrent('quick test loss',quicktest(learner,X_test,Y_test,samples_quicktest))

		if sc2.dispatch():
			trainer.save()

			fig1=getfnplot(sections,trainer.get_learned())
			cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)


			plotdata=cfg.sessionstate.hists
			cfg.remember('test loss',cfg.lossfn(trainer.get_learned()(X_test),Y_test))
			fig2=getlossplots(plotdata)
			cfg.savefig(*[path+'losses.pdf' for path in cfg.outpaths],fig=fig2)
			cfg.remember('NS norm',util.norm(learner.static_NS()(X_test)))
			cfg.remember('AS norm',util.norm(learner.as_static()(X_test)))
			fig3,fig4=getnormplots(plotdata)
			cfg.savefig(*[path+'fnorm.pdf' for path in cfg.outpaths],fig=fig3)
			cfg.savefig(*[path+'Afnorm.pdf' for path in cfg.outpaths],fig=fig4)

		

def quicktest(learner,X_test,Y_test,samples):
	I=np.random.choice(X_test.shape[0],samples)
	return cfg.lossfn(learner.as_static()(X_test[I]),Y_test[I])


#----------------------------------------------------------------------------------------------------


class E1plotter(pt.Plotter):pass
#
#	def processstate(weights):
#		



def getfnplot(sections,learned):
	plt.close('all')
	fig1,axs=plt.subplots(1,3,figsize=(16,4))
	sections.plot(axs,learned)
	return fig1





def getlossplots(plotdata):
	plt.close('all')
	fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))

	plotlosshist(ax21,plotdata)
	plotlosshist(ax22,plotdata)
	ax21.set_ylim(0,1)
	ax22.set_yscale('log')
	return fig2

def plotlosshist(ax,plotdata,logscale=False):
	train=plotdata['minibatch loss']
	test=plotdata['test loss']
	ax.plot(train['timestamps'],train['vals'],'r:',label='training loss')
	ax.plot(test['timestamps'],test['vals'],'bo-',label='test loss')
	
	ax.legend()
	ax.set_xlabel('seconds')





def getnormplots(plotdata):
	plt.close('all')
	fig1,(ax11,ax12)=plt.subplots(1,2,figsize=(15,7))
	fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))

	plotnormhist(ax11,ax21,plotdata) # f/A,A/f
	plotnormhist(ax12,ax22,plotdata) # f/A,A/f log plots
	ax12.set_yscale('log')
	ax22.set_yscale('log')
	return fig1,fig2

def plotnormhist(ax1,ax2,plotdata):
	NSnorm=plotdata['NS norm']
	ASnorm=plotdata['AS norm']

	ts,NSnorm,ASnorm=zip(*zip(NSnorm['timestamps'],NSnorm['vals'],ASnorm['vals']))

	Af_over_f=jnp.array(ASnorm)/jnp.array(NSnorm)

	ax1.plot(ts,NSnorm,'rd--',label='||f||')
	ax1.plot(ts,1/Af_over_f,'bo-',label='||f||/||Af||')

	ax2.plot(ts,ASnorm,'rd--',label='||Af||')
	ax2.plot(ts,Af_over_f,'bo-',label='||Af||/||f||')
	
	for ax in [ax1,ax2]:
		ax.legend()
		ax.set_xlabel('seconds')
		ax.grid(which='both')



#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1()
	run(sys.argv[1:])
