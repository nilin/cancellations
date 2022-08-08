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
'ftype':'AS_NN',
'n':7,
'd':1,
'samples_train':25000,
'samples_test':250,
'fnplotfineness':250,
'widths':[7,10,10,1],
'activation':'tanh',
'weight_decay':.1,
'lossfn':'SI_loss',
'samples_rademacher':100,
'timebound':60,
#'priorities':{'rademachercomplexity':1,'normalization':.01,'normratio':.01},
'priorities':{'rademachercomplexity':0,'normalization':1,'normratio':1},
'minibatchsize':50
}


def run():

	dashboard=cfg.dashboard
	cfg.trackduration=True

	if 'n' in cfg.cmdredefs:
		params['widths'][0]=cfg.cmdredefs['n']

	params.update(cfg.cmdredefs)
	globals().update(params)
	varnames=list(params)


	fnplotsched=cfg.stepwiseperiodicsched([1,5,10,60],[0,5,10,60,timebound])
	learningplotsched=cfg.stepwiseperiodicsched([5,10],[0,20,timebound])

	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	cfg.outpath='outputs/{}/{}/'.format(exname,cfg.sessionID)
	#cfg.outpath='outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID)
	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	session.remember('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,cfg.outpath+'info.txt',mode='w')


	#----------------------------------------------------------------------------------------------------


	dashboard.add_display(db.Display0(40,dashboard.width),0)
	dashboard.draw()


	#----------------------------------------------------------------------------------------------------
	

	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)


	#----------------------------------------------------------------------------------------------------

	initparams=(ftype,n,d,widths,activation)
	Af=ASf.init_learner(*initparams,lossfns=[])


	fig0=pt.singlefnplot_all_in_one(X_test,Af.as_static())
	cfg.savefig('{}{}'.format(cfg.outpath,'Af0.pdf'),fig=fig0)


	#----------------------------------------------------------------------------------------------------
	# round one
	# normalize Af, maximize ratio Af/f, let Af fit random points
	#----------------------------------------------------------------------------------------------------

	lognorm=lambda Y:jnp.log(jnp.average(Y**2))
	norm_one_loss=lambda Y:(jnp.log(jnp.average(Y**2)))**2

	normratiolossgrad=util.combinelossgradfns([mv.gen_lossgrad(Af.NS,lossfn=lognorm),mv.gen_lossgrad(Af.f,lossfn=lognorm)],[1,1],coefficients=[1,-1])
	normalizationlossgrad=mv.gen_lossgrad(Af.f,lossfn=norm_one_loss)	
	rademacherlossgrad=mv.gen_lossgrad(Af.f,lossfn=util.log_SI_loss)

	X_rademacher=rnd.uniform(cfg.nextkey(),(samples_rademacher,n,d),minval=-1,maxval=1)
	Y_rademacher=rnd.rademacher(cfg.nextkey(),(samples_rademacher,))

	lossgrad2=util.combinelossgradfns([rademacherlossgrad,normalizationlossgrad,normratiolossgrad],[2,1,1],[priorities[i] for i in ['rademachercomplexity','normalization','normratio']])	
	trainer=learning.Trainer(Af,X_rademacher,Y_rademacher,lossgrad=lossgrad2,weight_decay=weight_decay,minibatchsize=minibatchsize)


	processed=cfg.ActiveMemory()
	dashboard.add_display(Display1(10,dashboard.width,processed),40)
	sc1=cfg.Scheduler(cfg.expsched(.5,timebound))
	sc2=cfg.Scheduler(cfg.expsched(5,timebound))

	cfg.dblog(sc1.schedule)

	cfg.log('starting Af complexity optimization with concurrent normalization, Af/f optimization')

	i=0
	while trainer.memory.time()<timebound:
		try:
			loss=trainer.step()
			processed.remember('minibatch loss',loss)
			processed.addcontext('minibatch number',i)
			i=i+1
			processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in Af.weights[0]]))

			if sc1.activate():
				processed.remember('rademacher complexity',util.dot(Af.as_static()(X_rademacher),Y_rademacher))
				processed.remember('Af norm',jnp.average(Af.as_static()(X_test[:100])**2))
				processed.remember('f norm',jnp.average(Af.get_NS().as_static()(X_test[:100])**2))
				processed.compute(['f norm','Af norm'],lambda x,y:x/y,'f/Af')
				cfg.log('processed')

			if sc2.activate():
				plotexample(processed)

					
			session.trackcurrent('round2completeness',trainer.memory.time()/timebound)
		except KeyboardInterrupt:
			break


	cfg.log('complexity optimization with concurrent normalization, Af/f optimization done')
	
	fig1=pt.singlefnplot_all_in_one(X_test,Af.as_static())
	cfg.savefig('{}{}'.format(cfg.outpath,'Af1.pdf'),fig=fig1)




class Display1(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		
		#self.addnumberprint('rademacher complexity',msg='rademacher complexity estimate {:.3}')
		self.addnumberprint('Af norm',msg='||Af|| = {:.3}')
		self.addnumberprint('f norm',msg='||f|| = {:.3}')
		self.addnumberprint('f/Af',msg='||f||/||Af|| = {:.3}')
		self.addnumberprint('learner weightnorms',transform=util.norm,msg='weight norm = {:.3}')


def plotexample(memory):

	fig,(ax0,ax1)=plt.subplots(2,1,figsize=(7,9))

	#ax0.plot(*util.swap(*memory.gethist('AS norm','minibatch number')),label='AS norm')
	ax0.plot(*util.swap(*memory.gethist('f norm','minibatch number')),'b--',label='||f||')
	ax0.plot(*util.swap(*memory.gethist('f/Af','minibatch number')),'r-',label='||f||/||Af||')
	ax0.legend()
	ax0.set_yscale('log')
	#ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	weightnorms,minibatches=memory.gethist('learner weightnorms','minibatch number')
	for l,layer in enumerate(zip(*weightnorms)):
		ax1.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
	ax1.legend()
	ax1.set_ylim(bottom=0)
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)




	


if __name__=='__main__':

	cfg.dashboard=db.Dashboard0()
	run()
