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

import exRatio1,exRatio2

#jax.config.update("jax_enable_x64", True)


exname='exRatio'

explanation='Example '+exname#+': softplus target function'
#timebound=cfg.hour


params={
'n':5,
'd':1,
'samples_train':100000,
'samples_test':1000,
'fnplotfineness':250,
}|cfg.cmdredefs

params1={
'ftype':'AS_NN',
'widths':[5,50,50,1],
'weight_decay':.1,
'samples_rademacher':100,
'iterations1':1000,
'priorities':{'rademachercomplexity':1,'normratio':1,'normalization':1},
'minibatchsize':100
}

params2={
'targettype':'AS_NN',
'learnerwidths':[5,250,1],
'weight_decay':.1,
'iterations2':1000,
'minibatchsize':100
}

params1['widths'][0]=params['n']
params2['learnerwidths'][0]=params['n']

try:
	params1['activation']={'r':'ReLU','t':'tanh','d':'DReLU','p':'ptanh'}[cfg.selectone({'r','t','d','p'},cfg.cmdparams)]
except:
	print(10*'\n'+'Pass target activation function as parameter.\n'+10*'\n')	
	raise Exception


allparams=params|params1|params2
varnames_=[list(params),list(params1),list(params2)]


ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

sessioninfo='{}\nsessionID: {}\n\n{}\n\n{}\n\n{}'.format(explanation,cfg.sessionID,*[cfg.formatvars([(k,allparams[k]) for k in varnames],separator='\n',ignore=ignore) for varnames in varnames_])
for l in sessioninfo.splitlines():
	session.remember('sessioninfo',l)#.splitlines())





if __name__=='__main__':


	cfg.dashboard=db.Dashboard0()
	data=exRatio1.run(**allparams)
	db.clear()

	print('preparing round 2a, ReLU learner')
	cfg.dashboard=db.Dashboard0()
	exRatio2.run(**(allparams|{'learneractivation':'ReLU'}|data))
	db.clear()
	
	print('preparing round 2b, tanh learner')
	cfg.dashboard=db.Dashboard0()
	exRatio2.run(**(allparams|{'learneractivation':'tanh'}|data))
