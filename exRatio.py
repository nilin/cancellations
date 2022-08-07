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
from config import session


#jax.config.update("jax_enable_x64", True)


exname='exRatio'

explanation='Example '+exname#+': softplus target function'
#timebound=cfg.hour

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
'samples_rademacher':50,
'timebound':5,
#'round1iterations':25,
#'round2iterations':50,
'priorities':{'rademachercomplexity':1,'normalization':.01,'normratio':.01},
'minibatchsize':50
}
# does reach

#learningplotsched=cfg.stepwiseperiodicsched([5,10,60],[0,60,120,timebound])

def run():

	cfg.trackduration=True
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


	fnplotsched=cfg.stepwiseperiodicsched([1,5,10,60],[0,5,10,60,timebound])
	learningplotsched=cfg.stepwiseperiodicsched([5,10],[0,20,timebound])

	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	cfg.outpaths.add('outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID))
	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.log(sessioninfo)
	session.remember('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------

	
	
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)


	#----------------------------------------------------------------------------------------------------


	targetinitparams=(targettype,n,d,targetwidths,targetactivation)
	cfg.setlossfn(lossfn)
	target=ASf.init_learner(*targetinitparams)

	#----------------------------------------------------------------------------------------------------
	# round one, maximize ratio AS/NS of target
	#----------------------------------------------------------------------------------------------------



	cfg.log('optimizing AS/NS of target')


	fig0=pt.singlefnplot_all_in_one(X_test,target.as_static())
	cfg.savefig(*['{}{}'.format(path,'target0.pdf') for path in cfg.outpaths],fig=fig0)



#	target_ratio_trainer=learning.NoTargetTrainer(target,X,lossgrad=normratiolossgrad,weight_decay=weight_decay,minibatchsize=minibatchsize)
#	target_normalization_trainer=learning.NoTargetTrainer(target,X,lossgrad=normalizationlossgrad,weight_decay=weight_decay,minibatchsize=minibatchsize)
#
##	while target_ratio_trainer.timestamp()<round1time:
#	for i in range(round1iterations):
#		try:
#			target_ratio_trainer.step()
#			target_normalization_trainer.step()
#			cfg.trackcurrent('target AS norm',jnp.average(target.as_static()(X_test[:10])**2))
#			cfg.trackcurrent('target NS norm',jnp.average(target.get_NS().as_static()(X_test[:10])**2))
#
#			cfg.trackcurrent('round1completeness',i/round1iterations)
#			cfg.pokelisteners('refresh')
#		except KeyboardInterrupt:
#			break
#
#	cfg.log('first round of target AS/NS optimization done')
#
#	fig1=pt.singlefnplot_all_in_one(X_test,target.as_static())
#	cfg.savefig(*['{}{}'.format(path,'target1.pdf') for path in cfg.outpaths],fig=fig1)
#
	#----------------------------------------------------------------------------------------------------
	# round two, let target fit random points
	#----------------------------------------------------------------------------------------------------

	lognorm=lambda Y:jnp.log(jnp.average(Y**2))
	norm_one_loss=lambda Y:(jnp.log(jnp.average(Y**2)))**2

	normratiolossgrad=util.combinelossgradfns([mv.gen_lossgrad(target.NS,lossfn=lognorm),mv.gen_lossgrad(target.f,lossfn=lognorm)],[1,1],coefficients=[1,-1])
	normalizationlossgrad=mv.gen_lossgrad(target.f,lossfn=norm_one_loss)	
	rademacherlossgrad=mv.gen_lossgrad(target.f,lossfn=cfg.log_SI_loss)

	X_rademacher=rnd.uniform(cfg.nextkey(),(samples_rademacher,n,d),minval=-1,maxval=1)
	Y_rademacher=rnd.rademacher(cfg.nextkey(),(samples_rademacher,))

	lossgrad2=util.combinelossgradfns([rademacherlossgrad,normalizationlossgrad,normratiolossgrad],[2,1,1],[priorities[i] for i in ['rademachercomplexity','normalization','normratio']])	
	targettrainer=learning.Trainer(target,X_rademacher,Y_rademacher,lossgrad=lossgrad2,weight_decay=weight_decay,minibatchsize=minibatchsize)

	
	sc1=cfg.Scheduler(jnp.arange(timebound+1))

	while targettrainer.memory.time()<timebound:
	#for i in range(round2iterations):
		try:
			loss=targettrainer.step()

			if sc1.dispatch():
				cfg.trackcurrent('target rademacher complexity',cfg.dot(target.as_static()(X_rademacher),Y_rademacher))
				cfg.trackcurrent('target AS norm',jnp.average(target.as_static()(X_test[:100])**2))
				cfg.trackcurrent('target NS norm',jnp.average(target.get_NS().as_static()(X_test[:100])**2))

			cfg.pokelisteners('refresh')
			cfg.trackcurrent('round2completeness',targettrainer.memory.time()/timebound)
		except KeyboardInterrupt:
			break


	cfg.log('target complexity optimization with concurrent normalization, AS/NS optimization done')
	
	fig2=pt.singlefnplot_all_in_one(X_test,target.as_static())
	cfg.savefig(*['{}{}'.format(path,'target2.pdf') for path in cfg.outpaths],fig=fig2)


	#----------------------------------------------------------------------------------------------------
	# round 3, fix target and learn
	#----------------------------------------------------------------------------------------------------

	target=target.as_static()
	Y=target(X)
	
	cfg.setlossfn('SI_loss')
	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(*learnerinitparams)

	sc_save=cfg.Scheduler(cfg.stepwiseperiodicsched([1,10],[0,120,timebound]))
	sc_fnplot=cfg.Scheduler(fnplotsched)
	sc_learnplot=cfg.Scheduler(learningplotsched)

	learnertrainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay)



	sections=pt.CrossSections(X,Y,target,3,fineness=200)	

	while True:
		try:
			learnerloss=learnertrainer.step()
			cfg.trackcurrent('learner loss',learnerloss)
			cfg.pokelisteners('refresh')

			if sc_save.dispatch():
				learnertrainer.checkpoint()
				#cfg.autosave()


			if sc_fnplot.dispatch():
				#fig=pt.fnplot_all_in_one(X_test,target,learner.as_static(),normalized=False)

				fig=sections.getplot_SI(learner.as_static(),normalized=False)
				cfg.savefig(*['{}{}{}'.format(path,int(learnertrainer.memory.time()),' s.pdf') for path in cfg.outpaths],fig=fig)
				pass


				#reg_args=['learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation']
				#cfg.register(locals()|globals(),*reg_args)
				#dynamicplotter=pt.DynamicPlotter(locals()|globals(),reg_args,trainer.getlinks('minibatch loss','weights'))
#			if sc_learnplot.dispatch():
#				dynamicplotter.process_state(learner)
#				dynamicplotter.learningplots()
#				pass


		except KeyboardInterrupt:
			db.clear()			
			break




#


##----------------------------------------------------------------------------------------------------
#
#def prepdisplay():
#
#
#	slate=db.Display0()
#	#slate.trackvars('target AS norm','target NS norm','target rademacher complexity','learner loss','trainer loss','round1completeness','round2completeness')
#	#slate.memory.dynamicvals=[('target AS/NS',['target AS norm','target NS norm'],lambda x,y:x/y)]
#
#
#
#	slate.addline()
#	slate.addtext('round 1 progress')
#	slate.addbar('round1completeness')
#	slate.addtext('round 2 progress')
#	slate.addbar('round2completeness')
#
#	slate.addline()
#	slate.addtext(['target AS norm'],formatting=lambda x:'target norm {:.2}'.format(x[0]))
#	slate.addtext(['target NS norm'],formatting=lambda x:'target norm {:.2}'.format(x[0]))
#	slate.addtext(['target AS/NS'],formatting=lambda x:'target AS/NS {:.4f}'.format(x[0]))
#	slate.addtext(['target rademacher complexity'],formatting=lambda x:'target rademacher complexity {:.4f}'.format(x[0]))
#	slate.addspace()
#	slate.addline()
#	#slate.addvarprint('target fitrandom loss',formatting=lambda x:'target fitrandom loss {:.2}'.format(x))
#	#slate.addline()
#	slate.addtext(['learner loss'],formatting=lambda x:'learner loss {:.2f}'.format(x[0]))
#	slate.addbar(['learner loss'],avg_of=10)
#
#	return slate
#
#
#----------------------------------------------------------------------------------------------------

if __name__=='__main__':

	slate=prepdisplay()
	run()
