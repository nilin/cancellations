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
'n':6,
'd':1,
'samples_train':100000,
'samples_test':1000,
'fnplotfineness':250,
'weight_decay':.001,
}|cfg.cmdredefs

params1={
'ftype':'AS_NN',
'targetwidths':[6,10,10,1],
'samples_rademacher':100,
'iterations1':500,
'priorities':{'rademachercomplexity':1,'normratio':.5,'normalization':.5},
'minibatchsize':100
}

params2={
'targettype':'AS_NN',
'learnerwidths':[6,250,1],
'iterations2':10000,
'minibatchsize':100
}

params1['targetwidths'][0]=params['n']
params2['learnerwidths'][0]=params['n']


def plotexample(rmemory,tmemory):
	plt.close('all')

	fig,ax=plt.subplots(figsize=(6,4))
	ax.set_title('test losses')	

	ax.plot(*util.swap(*rmemory.gethist('test loss','minibatch number')),'b-',lw=2,label='ReLU learner')
	ax.plot(*util.swap(*tmemory.gethist('test loss','minibatch number')),'r--',lw=2,label='tanh learner')
	ax.legend()
	ax.set_ylim(bottom=0,top=1)
#	ax.grid(True,which='major',ls='-',axis='y')
#	ax.grid(True,which='minor',ls=':',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)

	fig,ax=plt.subplots(figsize=(6,4))
	ax.set_title('test losses')	

	ax.plot(*util.swap(*rmemory.gethist('test loss','minibatch number')),'b-',lw=2,label='ReLU learner')
	ax.plot(*util.swap(*tmemory.gethist('test loss','minibatch number')),'r--',lw=2,label='tanh learner')
	ax.legend()
	ax.set_yscale('log')
	ax.grid(True,which='major',ls='-',axis='y')
	ax.grid(True,which='minor',ls='-',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'loglosses.pdf'),fig=fig)

	fig,ax=plt.subplots(figsize=(6,4))
	ax.set_title('norm ratio ||f||/||Af|| (sign problem)')	

	ax.plot(*util.swap(*rmemory.gethist('f/Af','minibatch number')),'bo-',lw=1,label='ReLU learner')
	ax.plot(*util.swap(*tmemory.gethist('f/Af','minibatch number')),'rd--',lw=1,label='tanh learner')
	ax.legend()
	ax.set_yscale('log')
	ax.grid(True,which='major',ls='-',axis='y')
	#ax.grid(True,which='minor',ls=':',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'ratio.pdf'),fig=fig)



def run():

	try:
		params1['targetactivation']={'r':'ReLU','t':'tanh','d':'DReLU','p':'ptanh'}[cfg.selectone({'r','t','d','p'},cfg.cmdparams)]
	except:
		print(10*'\n'+'Pass target activation function as parameter.\n'+10*'\n')	
		raise Exception


	allparams=params|params1|params2
	varnames_=[list(params),list(params1),list(params2)]


	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	sessioninfo='{}\nsessionID: {}\n\n{}\n\n{}\n\n{}'.format(explanation,cfg.sessionID,*[cfg.formatvars([(k,allparams[k]) for k in varnames],separator='\n',ignore=ignore) for varnames in varnames_])
	for l in sessioninfo.splitlines():
		session.remember('sessioninfo',l)#.splitlines())
	session.remember('sessioninfostring',sessioninfo)



	outpath='outputs/{}/target={}/{}/'.format(exname,params1['targetactivation'],cfg.sessionID)
	cfg.write(session.getval('sessioninfostring'),outpath+'info.txt',mode='w')

	########################################################

	cfg.dbprint('round 1 prepare target')
	#cfg.dashboard=db.Dashboard0()
	data=exRatio1.run(**allparams)

	cfg.dbprint('round 2a ReLU learner')
	#cfg.dashboard=db.Dashboard0()
	rmem=exRatio2.run(**(allparams|{'learneractivation':'ReLU'}|data))
	
	cfg.dbprint('round 2b tanh learner')
	#cfg.dashboard=db.Dashboard0()
	tmem=exRatio2.run(**(allparams|{'learneractivation':'tanh'}|data))

	cfg.outpath=outpath
	plotexample(rmem,tmem)

	cfg.save({'r':rmem,'t':tmem},outpath+'data')

	db.clear()
	print('\noutputs in: outputs/exRatio2\n')



if __name__=='__main__':
	run()
