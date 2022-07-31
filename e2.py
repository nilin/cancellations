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
import e1


#jax.config.update("jax_enable_x64", True)



# e2
explanation='Example 2\n\
In order not to give an unfair advantage to either activation function \n\
we let the target function in this example be the sum of two antisymmetrized NNs, \n\
one constructed with each activation function. Both NNs are normalized to have the same magnitude.'


exname='e2'

params={
'targettype':'AS_NN',
'learnertype':'AS_NN',
'n':5,
'd':1,
'samples_train':10000,
'samples_test':1000,
'samples_quicktest':100,
'fnplotfineness':500,
'targetwidths':[5,100,1],
'learnerwidths':[5,100,1],
#'targetactivation':'tanh',
#'learneractivation':'ReLU',
'checkpoint_interval':5,
'timebound':cfg.hour
}



def run():

	# e2
	try:
		l_a={'r':'ReLU','relu':'ReLU','ReLU':'ReLU','t':'tanh','tanh':'tanh'}[cfg.cmdparams[0]]
	except:
		print(10*'\n'+'Pass activation function as first parameter.\n'+db.wideline()+10*'\n')	
		sys.exit(0)

	params['learneractivation']=l_a
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
	cfg.log('Generating AS functions.')

	targets=[ASf.init_target(targettype,n,d,targetwidths,ac) for ac in ['ReLU','tanh']]

	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,learneractivation)

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	# e2
	cfg.log('normalizing target terms')
	targets=[util.normalize(target,X[:100]) for target in targets]
	target=jax.jit(lambda X:targets[0](X)+targets[1](X))
	target=AS_HEAVY.makeblockwise(target)

	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner.as_static(),n,d)

	cfg.log('Generating training data Y.')
	Y=target(X)

	cfg.log('Generating test data Y.')
	Y_test=target(X_test)



	#
	sections=pt.CrossSections(X,Y,target,3,fineness=fnplotfineness)	

	reg_args=['learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation']
	cfg.register(locals()|globals(),*reg_args)
	dynamicplotter=e1.DynamicPlotter(locals()|globals(),reg_args,['minibatch loss','weights'])


	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	trainer=learning.Trainer(learner,X,Y)
	sc0=cfg.Scheduler(cfg.stepwiseperiodicsched([1,10],[0,120,timebound]))
	sc1=cfg.Scheduler(cfg.stepwiseperiodicsched([60],[0,timebound]))
	sc2=cfg.Scheduler(cfg.stepwiseperiodicsched([10],[0,timebound]))
	sc3=cfg.Scheduler(cfg.expsched(5,timebound,.2))
	sc4=cfg.Scheduler(cfg.stepwiseperiodicsched([5,30],[0,120,timebound]))
	cfg.log('\nStart training.\n')


	while True:
		try:
			trainer.step()
			cfg.pokelisteners('refresh')

			if sc0.dispatch():
				trainer.checkpoint()

			if sc1.dispatch():
				trainer.save()

			if sc2.dispatch():
				cfg.trackcurrent('quick test loss',e1.quicktest(learner,X_test,Y_test,samples_quicktest))

			if sc3.dispatch():
				"""
				fig1=e1.getfnplot(sections,learner.as_static())
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
				"""
				pass

			if sc4.dispatch():
				"""
				dynamicplotter.process_state(learner)
				dynamicplotter.learningplots()
				"""
				pass

		except KeyboardInterrupt:
			db.clear()			
			inp=input('Enter to continute, p+Enter to plot, q+Enter to end.\n')
			if inp=='p':
				fig1=e1.getfnplot(sections,learner.as_static())
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

				temp_plotter=e1.DynamicPlotter(locals()|globals(),reg_args,['minibatch loss','weights'])
				temp_plotter.prep(sc3.schedule)
				temp_plotter.learningplots()
				del temp_plotter
			if inp=='q':
				break
			db.clear()			

		


#----------------------------------------------------------------------------------------------------





#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1(params)

	cfg.trackduration=True
	run()
