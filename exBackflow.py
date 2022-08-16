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
import copy
import examplefunctions
import AS_functions as ASf
from config import session
import examples

#jax.config.update("jax_enable_x64", True)






cfg.exname='exBackflow'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':1,
'learnerwidths_b':[1,10,10,2],
'learnerwidths_f':[[2,9,9],[10,10,10],2],
'learneractivation':'tanh',
'targettype':'HermiteSlater',
'weight_decay':0,
'lossfn':'SI_loss',
'samples_train':100000,
'samples_test':1000,
'iterations':10000,
'minibatchsize':50
}

instructions='instructions:\n\npython exBackflow.py (b/f) \n\nparameters represent:\nbackflow+dets / ferminet\n'


def adjustparams():

	try:
		learnertype={'b':'backflowdets','f':'ferminet'}[cfg.selectone({'b','f'},cfg.cmdparams)]
		learnerwidths=cfg.params['learnerwidths_b'] if learnertype=='backflowdets' else cfg.params['learnerwidths_f']
	except:
		db.clear()
		print(instructions)
		quit()


	examples.adjustparams(learnertype=learnertype,learnerwidths=learnerwidths)



def run():
	globals().update(cfg.params)

	#global AS,NS,unprocessed,X,X_test,Y,Y_test,sections
	global AS,unprocessed,X,X_test,Y,Y_test,sections


	
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	target=ASf.init_target(targettype,n)
	Y=target(X)
	Y_test=target(X_test)

	#cfg.lossfn=util.SI_loss

	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(*learnerinitparams)
	#AS,NS=learner.AS,learner.NS
	AS=learner.AS

	trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize,lossgrad=mv.gen_lossgrad(AS,lossfn=util.SI_loss))

	unprocessed=cfg.ActiveMemory()
	try:
		cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,unprocessed),40,name='bars')
	except:
		pass

	sc1=cfg.Scheduler(cfg.stepwiseperiodicsched([10,100],[0,100,iterations]))
	sc2=cfg.Scheduler(cfg.sparsesched(iterations))

	sections=pt.CrossSections(X,Y,target)

	for i in range(iterations+1):

		cfg.poke()
		loss=trainer.step()

		unprocessed.addcontext('minibatchnumber',i)
		unprocessed.remember('minibatch loss',loss)

		if sc1.activate(i):
			unprocessed.remember('weights',learner.weights)

		if sc2.activate(i):
			#examples.processandplot(unprocessed,AS,NS,X_test,Y_test)
			processandplot(unprocessed,AS,X_test,Y_test)
			figtitle='learner activation {}'.format(learneractivation)
			figpath='{}{} {} minibatches'.format(cfg.outpath,learneractivation,int(unprocessed.getval('minibatchnumber')))
			examples.plotfunctions(sections,util.fixparams(AS,unprocessed.getval('weights')),figtitle,figpath)



def process_snapshot(processed,AS,weights,X,Y,i):
	processed.addcontext('minibatchnumber',i)

	AS=util.fixparams(AS,weights)

	#processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in weights[0]]))
	processed.remember('Af norm',jnp.average(AS(X[:100])**2))
	processed.remember('test loss',util.SI_loss(AS(X),Y))

	del AS


def processandplot(unprocessed,AS,X,Y,*args,**kw):
	processed=cfg.ActiveMemory()

	for weights,i in zip(*unprocessed.gethist('weights','minibatchnumber')):
		process_snapshot(processed,AS,weights,X,Y,i)		

	plotexample(unprocessed,processed,*args,**kw)
	cfg.save(processed,cfg.outpath+'data')
	return processed



def plotexample(unprocessed,processed,info=''):

	plt.close('all')

	fig,(ax0,ax1)=plt.subplots(2)
	fig.suptitle('test loss for learner '+info)

	ax0.plot(*util.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
	ax0.legend()
	ax0.set_ylim(bottom=0,top=1)
	ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	ax1.plot(*util.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
	ax1.legend()
	ax1.set_yscale('log')
	ax1.grid(True,which='major',ls='-',axis='y')
	ax1.grid(True,which='minor',ls=':',axis='y')

	cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)


#	fig,ax=plt.subplots()
#	ax.set_title('weight norms for learner '+info)
#
#	weightnorms,minibatches=processed.gethist('learner weightnorms','minibatchnumber')
#	for l,layer in enumerate(zip(*weightnorms)):
#		ax.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
#	ax.legend()
#	ax.set_ylim(bottom=0)
#	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)


	fig,ax=plt.subplots()
	ax.set_title('performance '+info)
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	cfg.savefig('{}{}'.format(cfg.outpath,'performance.pdf'),fig=fig)

	

class Display2(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
		#self.addbar('minibatch loss',style=db.dash)
		self.addbar('minibatch loss',style='.')
		self.addbar('minibatch loss',style=db.BOX,avg_of=50)
		self.addspace()
		self.addline()
		self.addnumberprint('minibatchnumber',msg='minibatch number {:.0f}/'+str(iterations))


def process_input(c):
	if c==108: #l
		#examples.processandplot(unprocessed,AS,NS,X_test,Y_test)
		processandplot(unprocessed,AS,X_test,Y_test)
	if c==102: #f
		figtitle='learner activation {}'.format(learneractivation)
		figpath='{}{} {} minibatches'.format(cfg.outpath,learneractivation,int(unprocessed.getval('minibatchnumber')))
		examples.plotfunctions(sections,util.fixparams(AS,unprocessed.getval('weights')),figtitle,figpath)



if __name__=='__main__':
	adjustparams()

	
	if 'display0' in cfg.cmdparams:
		cfg.dashboard=db.Dashboard0()
		run()
	else:
		import run_in_display
		run_in_display.RID(run,process_input)
