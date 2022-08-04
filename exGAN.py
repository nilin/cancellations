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
import multivariate as mv
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


exname='exGAN'

explanation='Example '+exname#+': softplus target function'
timebound=cfg.hour

params={
'targettype':'AS_NN',
'learnertype':'AS_NN',
'n':5,
'd':1,
'samples_train':25000,
'samples_test':250,
'fnplotfineness':250,
'targetwidths':[5,10,10,1],
'learnerwidths':[5,100,1],
'targetactivation':'tanh',
#'targetactivation':'DReLU',
#'learneractivation':'ReLU',
'weight_decay':.1,
'lossfn':'SI_loss',
'timebound':timebound
}
# does reach

fnplotsched=cfg.stepwiseperiodicsched([1,5,10,60],[0,5,10,60,timebound])
learningplotsched=cfg.stepwiseperiodicsched([5,10],[0,20,timebound])
#learningplotsched=cfg.stepwiseperiodicsched([5,10,60],[0,60,120,timebound])

def run():



	try:
		params['learneractivation']={'r':'ReLU','t':'tanh','s':'softplus','d':'DReLU'}[cfg.selectone({'r','t','s','d'},cfg.cmdparams)]
	except:
		print(10*'\n'+'Pass activation function as parameter.\n'+db.wideline()+10*'\n')	
		raise Exception

	if 'n' in cfg.cmdredefs:
		params['targetwidths'][0]=cfg.cmdredefs['n']
		params['learnerwidths'][0]=cfg.cmdredefs['n']

	params.update(cfg.cmdredefs)
	globals().update(params)
	varnames=list(params)



	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	cfg.outpaths.add('outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID))
	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.log(sessioninfo)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------

	
	
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)


	

	#----------------------------------------------------------------------------------------------------



	
	cfg.setlossfn('SI_loss')
	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(*learnerinitparams)




	sc_save=cfg.Scheduler(cfg.stepwiseperiodicsched([1,10],[0,120,timebound]))


	sc_fnplot=cfg.Scheduler(fnplotsched)
	sc_learnplot=cfg.Scheduler(learningplotsched)


	escapeloss=jax.jit(lambda Y1,Y2:-cfg.SI_loss(Y1,Y2))
	normloss=lambda Y1:(1-jnp.average(Y1**2))**2

	cfg.lossfn=escapeloss
	targetinitparams=(targettype,n,d,targetwidths,targetactivation)
	target=ASf.init_learner(*targetinitparams)


	learnertrainer=learning.DynamicTrainer(learner,X,weight_decay=weight_decay)
	targettrainer=learning.DynamicTrainer(target,X,weight_decay=weight_decay)
	targetnormtrainer=learning.NoTargetTrainer(target,X,lossgrad=mv.gen_lossgrad(target.f,lossfn=normloss),weight_decay=weight_decay)

	learnersteps=1
	targetsteps=1
	targetnormsteps=1

	while True:
		try:


			for i in range(learnersteps):			
				learnerloss=learnertrainer.step(target.as_static())
				cfg.trackcurrent('learner loss',learnerloss)

			for i in range(targetsteps):
				trainerloss=targettrainer.step(learner.as_static())
				cfg.trackcurrent('trainer loss',trainerloss)

			for i in range(targetnormsteps):
				targetnormtrainer.step()
				cfg.trackcurrent('target norm',jnp.average(target.as_static()(X_test[:10])**2))
			

			cfg.pokelisteners('refresh')



			if sc_save.dispatch():
				learnertrainer.checkpoint()
				targettrainer.checkpoint()
				cfg.autosave()


			if sc_fnplot.dispatch():

				_target_=AS_HEAVY.makeblockwise(target.as_static())
				Y_test=_target_(X_test)
				sections=pt.CrossSections(X_test,Y_test,_target_,3,fineness=fnplotfineness)	
				#reg_args=['learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation']
				#cfg.register(locals()|globals(),*reg_args)
				#dynamicplotter=pt.DynamicPlotter(locals()|globals(),reg_args,trainer.getlinks('minibatch loss','weights'))

				nlrn=util.closest_multiple(learner.as_static(),X_test[:500],Y_test[:500],normalized=True)
				fig1=pt.getfnplot(sections,nlrn)
				cfg.savefig(*['{}{}{}'.format(path,int(sc_save.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
				del sections

#
#			if sc_learnplot.dispatch():
#				dynamicplotter.process_state(learner)
#				dynamicplotter.learningplots()
#				pass


		except KeyboardInterrupt:
			db.clear()			
			learnersteps=int(input('Enter learner steps.\n'))
			targetsteps=int(input('Enter target steps.\n'))
			db.clear()			






#----------------------------------------------------------------------------------------------------

def prepdisplay():


	slate=db.Display0()
	slate.trackvars('target norm','learner loss','trainer loss')

	slate.addvarprint('target norm',formatting=lambda x:'target norm {:.10f}'.format(x))
	slate.addline()
	slate.addvarprint('learner loss',formatting=lambda x:'learner loss {:.2f}'.format(x))
	slate.addbar('learner loss',avg_of=10)
	slate.addvarprint('trainer loss',formatting=lambda x:'trainer loss {:.2f}'.format(x))
	slate.addbar('trainer loss',transform=lambda x:-x,style=' ',emptystyle=cfg.BOX,avg_of=10)

	return slate


#----------------------------------------------------------------------------------------------------


def runwithdisplay():
	slate=prepdisplay()
	cfg.trackduration=True
	run()


if __name__=='__main__':
	runwithdisplay()
