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
import e1

#jax.config.update("jax_enable_x64", True)




# e2
explanation='Example 2\n\
In order not to give an unfair advantage to either activation function \n\
we let the target function in this example be the sum of two antisymmetrized NNs, \n\
one constructed with each activation function. Both NNs are normalized to have the same magnitude.'


exname='e2'


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
	#'targetactivation':'tanh',
	#'learneractivation':'ReLU',
	'checkpoint_interval':5,
	'timebound':cfg.hour
	}
	args,redefs=cfg.parse_cmdln_args(cmdargs)



	# e2
	try:
		l_a={'r':'ReLU','relu':'ReLU','ReLU':'ReLU','t':'tanh','tanh':'tanh'}[args[0]]
	except:
		print(10*'\n'+'Pass activation function as first parameter.\n'+db.wideline()+10*'\n')	
		sys.exit(0)

	params['learneractivation']=l_a

	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)


	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','d','checkpoint_interval'}

	cfg.outpaths.add('outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID))
	sessioninfo=explanation+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
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

	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner.as_static(),n,d)

	cfg.log('Generating data Y.')
	Y=target(X)
	Y_test=target(X_test)



	#
	sections=pt.CrossSections(X,Y,target,3)	
	cfg.register(locals(),'learnerinitparams','X','Y','X_test','Y_test','sections')
	plotter=Plotter(['X_test','Y_test'],['minibatch loss'])

	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	


	trainer=learning.Trainer(learner,X,Y)
	sc0=cfg.Scheduler(cfg.expsched(.25,timebound))
	sc1=cfg.Scheduler(cfg.periodicsched(5,timebound))
	sc2=cfg.Scheduler(cfg.periodicsched(1,timebound))
	cfg.log('\nStart training.\n')



	while True:
		trainer.step()
		cfg.pokelisteners('refresh')

		if sc0.dispatch():
			trainer.checkpoint()

		if sc1.dispatch():
			cfg.trackcurrent('quick test loss',e1.quicktest(learner,X_test,Y_test,samples_quicktest))

		if sc2.dispatch():
			trainer.save()

			fig1=e1.getfnplot(sections,learner.as_static())
			cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

			plotter.process_state(learner)
			plotter.plotlosshist()
			plotter.plotweightnorms()
			plotter.plot3()

		


#----------------------------------------------------------------------------------------------------


class Plotter(e1.Plotter):pass


#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1()
	run(sys.argv[1:])
