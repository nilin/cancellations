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
import functions
from config import session
import examples

#jax.config.update("jax_enable_x64", True)






cfg.exname='ASNN_learn_slater'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':2,
'learnertype':'AS_NN',
'learnerwidths':[10,100,100,1],
#'learneractivation':'tanh',
#'targettype':'hermiteSlater',
'weight_decay':0,
'lossfn':'SI_loss',
'samples_train':50000,
'samples_test':1000,
'iterations':100000,
'minibatchsize':50
}

instructions='instructions:\n\n\
python e_ASNN_learn_slater.py (t/r/lr/s) (h/g) \n\n\
parameters represent:\n\
tanh/relu/leaky relu/softplus learner\n\
hermite/gaussian slater target\n'


try:
	examples.adjustparams(learneractivation=cfg.fromcmdparams(t='tanh',s='softplus',r='relu',lr='leakyrelu'),targettype=cfg.fromcmdparams(h='hermite',g='gaussian')+'Slater')
except:
	db.clear()
	print(instructions)
	quit()

globals().update(cfg.params)
target=functions.StaticFunc(ftype=targettype,n=n,d=d)
learner=functions.DynFunc(ftype=learnertype,n=n,d=d,widths=learnerwidths,activation=learneractivation)
examples.runexample(target,learner)
