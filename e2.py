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
from pynput import keyboard
import testing
import AS_tools
import AS_HEAVY
import examplefunctions
import AS_functions as ASf


#jax.config.update("jax_enable_x64", True)


exname='e2'

explanation='Train a network with activation fn #1 to fit one with activation #2'



params={
'targettype':'AS_NN',
'learnertype':'AS_NN',
'n':7,
'd':1,
'samples_train':2500,
'samples_test':250,
'fnplotfineness':250,
'targetwidths':[7,100,1],
'learnerwidths':[7,250,1],
#'targetactivation':'tanh',
#'learneractivation':'ReLU',
'timebound':cfg.hour
}
# does reach


def run():



	try:
		params['learneractivation'],params['targetactivation']=[{'r':'ReLU','t':'tanh'}[k] for k in cfg.cmdparams[-2:]]
	except:
		print(10*'\n'+'Pass activation functions as parameters (learner, target).\n'+db.wideline()+10*'\n')	
		raise Exception

	if 'n' in cfg.cmdredefs:
		params['targetwidths'][0]=cfg.cmdredefs['n']
		params['learnerwidths'][0]=cfg.cmdredefs['n']

	globals().update(params)
	globals().update(cfg.cmdredefs)
	varnames=cfg.orderedunion(params,cfg.cmdredefs)


	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	cfg.outpaths.add('outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID))
	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------
	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,learneractivation)

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	#targets=[ASf.init_target(targettype,n,d,targetwidths,ac) for ac in ['ReLU','tanh']]
	#cfg.log('normalizing target terms')
	#targets=[util.normalize(target,X[:100]) for target in targets]
	#target=jax.jit(lambda X:targets[0](X)+targets[1](X))
	#target=AS_HEAVY.makeblockwise(target)

	target=ASf.init_target(targettype,n,d,targetwidths,targetactivation)
	cfg.log('normalizing target')
	target=util.normalize(target,X[:100])
	target=AS_HEAVY.makeblockwise(target)



	#----------------------------------------------------------------------------------------------------
	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner.as_static(),n,d)

	cfg.log('Generating training data Y.')
	Y=target(X)

	cfg.log('Generating test data Y.')
	Y_test=target(X_test)



	trainer=learning.Trainer(learner,X,Y)
	sections=pt.CrossSections(X,Y,target,3,fineness=fnplotfineness)	
	reg_args=['learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation']
	cfg.register(locals()|globals(),*reg_args)
	dynamicplotter=pt.DynamicPlotter(locals()|globals(),reg_args,trainer.getlinks('minibatch loss','weights'))
	

	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	sc0=cfg.Scheduler(cfg.stepwiseperiodicsched([1,10],[0,120,timebound]))
	sc1=cfg.Scheduler(cfg.stepwiseperiodicsched([60],[0,timebound]))
	#sc2=cfg.Scheduler(cfg.stepwiseperiodicsched([10],[0,timebound]))
	#sc3=cfg.Scheduler(cfg.stepwiseperiodicsched([5,30],[0,120,timebound]))
	sc4=cfg.Scheduler(cfg.expsched(2,timebound,.1))
	cfg.log('\nStart training.\n')

	
	while True:
		try:
			trainer.step()
			cfg.pokelisteners('refresh')

			if sc0.dispatch():
				trainer.checkpoint()

			if sc1.dispatch():
				trainer.save()

			#if sc2.dispatch():
			#	"""
			#	cfg.trackcurrent('quick test loss',e0.quicktest(learner,X_test,Y_test,samples_quicktest))
			#	"""

			#if sc3.dispatch():
			#	"""
			#	fig1=pt.getfnplot(sections,learner.as_static())
			#	cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
			#	"""
			#	pass

			#if sc4.dispatch():
			#	"""
			#	dynamicplotter.process_state(learner)
			#	dynamicplotter.learningplots()
			#	"""
			#	pass


		except KeyboardInterrupt:
			db.clear()			
			inp=input('Enter to continute, p+Enter to plot, q+Enter to end.\n')
			if inp=='p':
				fig1=pt.getfnplot(sections,learner.as_static())
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

				temp_plotter=pt.DynamicPlotter(locals()|globals(),reg_args,trainer.getlinks('minibatch loss','weights'))
				temp_plotter.prep(sc4.schedule)
				temp_plotter.learningplots()
				del temp_plotter
			if inp=='q': break
			db.clear()			
		


#----------------------------------------------------------------------------------------------------





#----------------------------------------------------------------------------------------------------


def main():
	slate=db.display_1()
	cfg.trackduration=True
	run()


if __name__=='__main__':
	main()
