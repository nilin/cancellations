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
import multivariate as mv
import AS_functions as ASf


#jax.config.update("jax_enable_x64", True)


exname='e3'

explanation='Example '+exname#+': softplus target function'
timebound=cfg.hour

params={
#'targettype':'AS_NN',
#'learnertype':'AS_NN',
#'learningmode':'AS',
'n':4,
'd':1,
'samples_train':25000,
'samples_test':250,
'fnplotfineness':250,
'targetwidths':[4,1,1],
'learnerwidths':[4,100,1],
'targetactivation':'tanh',
#'targetactivation':'DReLU',
#'learneractivation':'ReLU',
'weight_decay':.01,
'lossfn':'SI_loss',
'randseed':0,
'timebound':timebound
}
# does reach

fnplotsched=cfg.stepwiseperiodicsched([1,5,10,60],[0,5,10,60,timebound])
learningplotsched=cfg.stepwiseperiodicsched([5,10],[0,20,timebound])
#learningplotsched=cfg.stepwiseperiodicsched([5,10,60],[0,60,120,timebound])

def run():


	try:
		params['learneractivation']={'r':'ReLU','t':'tanh'}[cfg.selectone({'r','t'},cfg.cmdparams)]
		params['learningmode']=cfg.selectone({'AS','NS'},cfg.cmdparams)
	except:
		print(10*'\n'+'Pass activation function as parameter.\n'+db.wideline()+10*'\n')	
		raise Exception

	if 'n' in cfg.cmdredefs:
		params['targetwidths'][0]=cfg.cmdredefs['n']
		params['learnerwidths'][0]=cfg.cmdredefs['n']

	params.update(cfg.cmdredefs)
	globals().update(params)
	varnames=list(params)


	cfg.keychain.resetkeys(randseed)
	cfg.setlossfn(lossfn)


	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	cfg.outpaths.add('outputs/{}/{} {}/{}/'.format(exname,learningmode,learneractivation,cfg.sessionID))
	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.log(sessioninfo)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------
	lNN=mv.gen_NN_NS(learneractivation)
	learnerweights=mv.initweights_NN(learnerwidths)
	Af=AS_tools.gen_Af(n,lNN)
	learner_AS=learning.AS_Learner(Af,lossgrad=AS_tools.gen_lossgrad_Af(n,lNN),NS=lNN,weights=learnerweights,deepcopy=False)
	learner_NS=learning.NS_Learner(lNN,weights=learnerweights,deepcopy=False)

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	targetweights=mv.initweights_NN(targetwidths)
	tNN=mv.gen_NN_NS(targetactivation)
	target_NS,target_AS=util.fixparams(tNN,targetweights),util.fixparams(AS_tools.gen_Af(n,tNN),targetweights)
	cfg.log('normalizing targets')
	target_NS=util.normalize(target_NS,X[:500],echo=True)
	target_AS=util.normalize(target_AS,X[:500],echo=True)
	target_AS=AS_HEAVY.makeblockwise(target_AS)


	#----------------------------------------------------------------------------------------------------
	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target_AS,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner_AS.as_static(),n,d)


	target=locals()['target_'+learningmode]
	learner=locals()['learner_'+learningmode]

	cfg.log('Generating training data Y.')
	Y=target(X)

	cfg.log('Generating test data Y.')
	Y_test=target_AS(X_test)



	trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay)#,minibatchsize=500)

	sections=pt.CrossSections(X,Y,target_AS,3,fineness=fnplotfineness)	
	learnerinitparams=('AS_NN',n,d,learnerwidths,learneractivation)
	reg_args=['X','Y','X_test','Y_test','sections','learneractivation','learnerinitparams']
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
				nlrn=util.closest_multiple(util.fixparams(Af,learner.weights),X_test[:500],Y_test[:500],normalized=True)
				fig1=pt.getfnplot(sections,nlrn)
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
				pass

			if sc_learnplot.dispatch():
				dynamicplotter.process_state(learning.AS_Learner(Af,NS=lNN,weights=learner.cloneweights()))
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
