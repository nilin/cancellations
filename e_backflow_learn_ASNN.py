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
import functions
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
from config import session
import examples

#jax.config.update("jax_enable_x64", True)






cfg.exname='backflow_learn_ASNN'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':2,
#'learnertype':'AS_NN',
#'learnerwidths':[10,100,100,1],
'learnerwidths_b':[[2,5,25],5],
'learnerwidths_p':[[2,5,100],5],
'learnerwidths_f':[[3,8,8],[10,10,10],2],
#'learneractivation':'tanh',
'targettype':'AS_NN',
'targetwidths':[10,20,20,1],
'targetactivation':'tanh',
'weight_decay':0,
'lossfn':'SI_loss',
'samples_train':25000,
'samples_test':1000,
'iterations':100000,
'minibatchsize':50
}

#instructions='instructions:\n\npython e_backflow_learn_ASNN.py (b/f/p) \n\n\
#parameters represent:\nb=backflow+dets / f=ferminet / p=backflow_detsandsym (product of sym and det)\n'

instructions='instructions:\n\npython e_backflow_learn_ASNN.py (t/lr) \n\n\
parameters represent:\ntanh/leaky relu\n'


try:
	selection='p' #cfg.selectone({'b','f','p'},cfg.cmdparams)
	#selection=cfg.selectone({'b','f','p'},cfg.cmdparams)
	learnertype={'b':'backflowdets','f':'ferminet','p':'backflow_detsandsym'}[selection]
	learnerwidths=cfg.params['learnerwidths_'+selection]
	examples.adjustparams(learnertype=learnertype,learnerwidths=learnerwidths,learneractivation=cfg.fromcmdparams(t='tanh',lr='leakyrelu'))
except:
	db.clear()
	print(instructions)
	quit()
globals().update(cfg.params)

target=functions.DynFunc(ftype=targettype,n=n,d=d,widths=targetwidths,activation=targetactivation)
learner=functions.DynFunc(ftype=learnertype,n=n,d=d,widths=learnerwidths,activation=learneractivation)
examples.runexample(target,learner)
