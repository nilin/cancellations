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






cfg.exname='ASNN_learn_ASNN'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':2,
'learnertype':'AS_NN',
'learnerwidths':[10,100,100,1],
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

instructions='instructions:\n\n\
python e_ASNN_learn_ASNN.py (t/r/lr/s) \n\n\
parameters represent:\n\
tanh/relu/leaky relu/softplus learner\n'


def process_snapshot_1(processed,dynfunc,X,Y,i):
	examples.process_snapshot_0(processed,dynfunc,X,Y,i)
	processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in dynfunc.weights[0]]))

def plotexample_1(unprocessed,processed,info=''):
	examples.plotexample_0(unprocessed,processed,info)

	fig,ax=plt.subplots()
	ax.set_title('weight norms for learner '+info)

	weightnorms,minibatches=processed.gethist('learner weightnorms','minibatchnumber')
	for l,layer in enumerate(zip(*weightnorms)):
		ax.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
	ax.legend()
	ax.set_ylim(bottom=0)
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)


examples.process_snapshot=process_snapshot_1
examples.plotexample=plotexample_1


try:
	examples.adjustparams(learneractivation=cfg.getfromcmdparams(t='tanh',r='relu',s='softplus',lr='leakyrelu'))
	pass
except:
	db.clear()
	print(instructions)
	quit()

globals().update(cfg.params)
target=functions.DynFunc(ftype=targettype,n=n,d=d,widths=targetwidths,activation=targetactivation)
learner=functions.DynFunc(ftype=learnertype,n=n,d=d,widths=learnerwidths,activation=learneractivation)

examples.runexample(target,learner)
