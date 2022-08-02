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


exname='e1'

explanation='Example '+exname+'\n\
In order not to give an unfair advantage to either activation function \n\
we let the target function in this example be the sum of two antisymmetrized NNs, \n\
one constructed with each activation function. Both NNs are normalized to have the same magnitude.'

timebound=cfg.hour

params={
'targettype':'AS_NN',
'learnertype':'AS_NN',
'n':6,
'd':1,
'samples_train':25000,
'samples_test':1000,
'fnplotfineness':500,
'targetwidths':[6,100,100,1],
'learnerwidths':[6,250,1],
#'targetactivation':'tanh',
#'learneractivation':'ReLU',
'lossfn':'SI_loss',
'timebound':timebound
}
# does reach

fnplotsched=cfg.stepwiseperiodicsched([5,10,60],[0,60,120,timebound])
learningplotsched=cfg.stepwiseperiodicsched([5,10,60],[0,60,120,timebound])

def run():



	try:
		l_a={'r':'ReLU','t':'tanh'}[cfg.selectone({'r','t'},cfg.cmdparams)]
	except:
		print(10*'\n'+'Pass activation function as parameter.\n'+db.wideline()+10*'\n')	
		raise Exception

	params['learneractivation']=l_a
	if 'n' in cfg.cmdredefs:
		params['targetwidths'][0]=cfg.cmdredefs['n']
		params['learnerwidths'][0]=cfg.cmdredefs['n']

	globals().update(params)
	globals().update(cfg.cmdredefs)
	varnames=cfg.orderedunion(params,cfg.cmdredefs)

	cfg.setlossfn(lossfn)

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

	targets=[ASf.init_target(targettype,n,d,targetwidths,ac)[0] for ac in ['DReLU','tanh']]
	cfg.log('normalizing target terms')
	targets=[util.normalize(target,X[:1000]) for target in targets]
	target=jax.jit(lambda X:targets[0](X)+targets[1](X))
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
	sc1=cfg.Scheduler(cfg.stepwiseperiodicsched([15],[0,timebound]))


	sc_fnplot=cfg.Scheduler(fnplotsched)
	sc_learnplot=cfg.Scheduler(learningplotsched)

	cfg.log('\nStart training.\n')

	
	while True:
		try:
			trainer.step()
			cfg.pokelisteners('refresh')

			if sc0.dispatch():
				trainer.checkpoint()

			if sc1.dispatch():
				trainer.save()

			if sc_fnplot.dispatch():
				nlrn=util.closest_multiple(learner.as_static(),X_test[:500],Y_test[:500],normalized=True)
				fig1=pt.getfnplot(sections,nlrn)
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
				pass

			if sc_learnplot.dispatch():
				dynamicplotter.process_state(learner)
				dynamicplotter.learningplots()
				pass


		except KeyboardInterrupt:
			db.clear()			
			inp=input('Enter to continute, p+Enter to plot, q+Enter to end.\n')
			if inp=='p':
				fig1=pt.getfnplot(sections,learner.as_static())
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

				temp_plotter=pt.DynamicPlotter(locals()|globals(),reg_args,trainer.getlinks('minibatch loss','weights'))
				temp_plotter.prep(learningplotsched)
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
