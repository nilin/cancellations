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
	'samples_test':1000,
	'targetwidths':[5,25,25,1],
	'learnerwidths':[5,100,1],
	'targetactivation':'tanh',
	'learneractivation':'ReLU',
	'checkpoint_interval':5,
	'timebound':600
	}


	_,redefs=cfg.parse_cmdln_args(cmdargs)
	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)


	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','d','checkpoint_interval'}
	if 'NN' not in targettype: ignore.update({'targetwidths','targetactivation'})

	sessioninfo=explanation+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.log('sessioninfo:\n'+sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------
	cfg.log('Generating AS functions.')

	t_args=(n,d,targetwidths,activations[targetactivation]) if 'NN' in targettype else (n,)
	target=ASf.init_target(targettype,*t_args) 
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,activations[learneractivation])

	
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
	cfg.log('prepare cross sections for plotting')
	sections=pt.CrossSections(X,Y,target,3)

	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------


	trainer=learning.Trainer(learner,X,Y)
	sc1=cfg.Scheduler(cfg.defaultsched)
	sc2=cfg.Scheduler(cfg.arange(0,3600,5)+cfg.arange(3600,3600*24,3600))
	cfg.log('\nStart training.\n')

	while time.perf_counter()<timebound:
		trainer.step()
		cfg.pokelisteners('refresh')

		if sc1.dispatch():
			trainer.save()
			fig1=getfnplot(trainer.get_learned(),target,X_test,Y_test)
			cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

		if sc2.dispatch():
			cfg.trackhist('test loss',cfg.lossfn(trainer.get_learned()(X_test),Y_test))
			fig2=getlossplots()
			cfg.savefig(*[path+'losses.pdf' for path in cfg.outpaths],fig=fig2)






#----------------------------------------------------------------------------------------------------


def getlossplots():
	plt.close('all')
	fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))

	plotlosshist(ax21,cfg.gethists())
	plotlosshist(ax22,cfg.gethists(),logscale=True)
	return fig2


def plotlosshist(ax,hists,logscale=False):
	train=hists['minibatch loss']
	test=hists['test loss']
	ax.plot(train['timestamps'],train['vals'],'r:',label='training loss')
	ax.plot(test['timestamps'],test['vals'],'bo-',label='test loss')
	
	ax.legend()
	ax.set_xlabel('seconds')
	if logscale:
		ax.set_yscale('log')
	else:
		ax.set_ylim(0,1)


def getfnplot(learned,target,X_test,Y_test):
	plt.close('all')
	fig1,axs=plt.subplots(1,3,figsize=(15,5))
	pt.plotalongline(ax1,learned,target,X_test,Y_test)
	return fig1

#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	db.display_1()
	run(sys.argv[1:])
