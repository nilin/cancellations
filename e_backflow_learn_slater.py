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






cfg.exname='backflow_learn_slater'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':2,
'learnertype':'backflow_detsandsym',
'learnerwidths':[[2,10,100],10],
#'learnertype':'AS_NN',
#'learnerwidths':[10,250,1],
'learneractivation':'tanh',
#'targettype':'gaussianSlater',
'weight_decay':0,
'lossfn':'SI_loss',
'samples_train':50000,
'samples_test':500,
'iterations':100000,
'minibatchsize':100
}

instructions='instructions:\n\n\
python e_backflow_learn_slater.py (h/g) (t/lr) \n\n\
parameters represent:\n\
h=hermite/g=gaussian slater target\n\
tanh/leaky relu learner\n'


try:
	examples.adjustparams(targettype=cfg.getfromcmdparams(h='hermite',g='gaussian')+'Slater',learneractivation=cfg.getfromcmdparams(t='tanh',lr='leakyrelu'))
	pass
except:
	db.clear()
	print(instructions)
	quit()
globals().update(cfg.params)

target=functions.StaticFunc(ftype=targettype,n=n,d=d)
learner=functions.DynFunc(ftype=learnertype,n=n,d=d,widths=learnerwidths,activation=learneractivation)
examples.runexample(target,learner)
