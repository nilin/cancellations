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






cfg.exname='exSlater'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':1,
'targettype':'HermiteSlater',
'learnerwidths':[None,250,1],
'weight_decay':.1,
'lossfn':'SI_loss',
'samples_train':100000,
'samples_test':1000,
'iterations':10000,
'minibatchsize':50
}

instructions='instructions:\n\npython exSlater.py (r/t) (decay/increase/none) (H/G)\n\nparameters represent:\nrelu/tanh\nweights\nHermite Slater/Gaussian Slater\n\norder of arguments is not fixed\n'


def adjustparams():

	try:
		cfg.params['learneractivation']={'r':'ReLU','t':'tanh','d':'DReLU','p':'ptanh'}[cfg.selectone({'r','t','d','p'},cfg.cmdparams)]
		cfg.params['weight_decay']={'decay':.1,'increase':-.1,'none':0}[cfg.selectone({'decay','increase','none'},cfg.cmdparams)]
		cfg.params['targettype']={'H':'HermiteSlater','G':'GaussianSlater1D'}[cfg.selectone({'H','G'},cfg.cmdparams)]
	except:
		db.clear()
		print(instructions)
		quit()


	examples.adjustparams()



def run():
	globals().update(cfg.params)

	global AS,NS,unprocessed,X,X_test,Y,Y_test,sections


	
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	target=ASf.init_target(targettype,n)

	Y=target(X)
	Y_test=target(X_test)


	cfg.lossfn=util.SI_loss
	learnerinitparams=('AS_NN',n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(*learnerinitparams)
	AS,NS=learner.AS,learner.NS

	trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize)

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
			examples.processandplot(unprocessed,AS,NS,X_test,Y_test)
			figtitle='learner activation {}'.format(learneractivation)
			figpath='{}{} {} minibatches'.format(cfg.outpath,learneractivation,int(unprocessed.getval('minibatchnumber')))
			examples.plotfunctions(sections,util.fixparams(AS,unprocessed.getval('weights')),figtitle,figpath)



	

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
		examples.processandplot(unprocessed,AS,NS,X_test,Y_test)
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
