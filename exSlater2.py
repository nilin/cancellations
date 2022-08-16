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


#jax.config.update("jax_enable_x64", True)





from examples import Example




class ExSlater(Example):

	def __init__(self):
		self.exname='exSlater'

		self.params={
		'ftype':'AS_NN',
		'n':5,
		'd':1,
		'targettype':'HermiteSlater',
		'learnerwidths':[5,250,1],
		'weight_decay':.1,
		'lossfn':'SI_loss',
		'samples_train':100000,
		'samples_test':1000,
		'iterations':1000,
		'minibatchsize':50
		}

		super().__init__()

		target=ASf.init_target(targettype,n)
		Y=target(X)
		Y_test=target(X_test)

		cfg.lossfn=util.SI_loss
		learnerinitparams=(ftype,n,d,learnerwidths,learneractivation)
		learner=ASf.init_learner(*learnerinitparams)
		AS,NS=learner.AS,learner.NS

		trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize)
		sections=pt.CrossSections(X,Y,target)

	def prep(self):
		cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,unprocessed),40,name='bars')


	

class Display2(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
		self.addbar('minibatch loss',style=db.dash)
		self.addbar('minibatch loss',style=db.BOX,avg_of=25)
		self.addspace()
		#self.addnumberprint('test loss',msg='test loss {:.3}')
		#self.addbar('test loss')
		self.addline()
		self.addnumberprint('minibatchnumber',msg='minibatch number {:.0f}/'+str(iterations))




if __name__=='__main__':

	import run_in_display
	e=ExSlater()
	run_in_display.classRID(e)
