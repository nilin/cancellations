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




def adjustparams(**priorityparams):

	cfg.params=priorityparams|cfg.params
	params=cfg.params
	explanation=cfg.explanation
	params.update(cfg.cmdredefs)

	#params['learnerwidths'][0]=params['n']*params['d']

	info='\n'.join(['{}={}'.format(k,v) for k,v in params.items()])
	sessioninfo='{}\nsessionID: {}\n\n{}'.format(explanation,cfg.sessionID,info)
	session.remember('sessioninfo',sessioninfo)

	cfg.trackduration=True
	cfg.outpath='outputs/{}/{}/'.format(cfg.exname,cfg.sessionID)
	cfg.write(session.getval('sessioninfo'),cfg.outpath+'info.txt',mode='w')





def process_snapshot(processed,AS,NS,weights,X,Y,i):
	processed.addcontext('minibatchnumber',i)

	AS=util.fixparams(AS,weights)
	NS=util.fixparams(NS,weights)

	processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in weights[0]]))
	processed.remember('Af norm',jnp.average(AS(X[:100])**2))
	processed.remember('f norm',jnp.average(NS(X[:100])**2))
	processed.compute(['f norm','Af norm'],lambda x,y:x/y,'f/Af')
	processed.remember('test loss',util.SI_loss(AS(X),Y))

	del AS,NS


def processandplot(unprocessed,AS,NS,X,Y,*args,**kw):
	processed=cfg.ActiveMemory()

	for weights,i in zip(*unprocessed.gethist('weights','minibatchnumber')):
		process_snapshot(processed,AS,NS,weights,X,Y,i)		

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


	fig,ax=plt.subplots()
	ax.set_title('norm ratio (sign problem) for learner '+info)

	ax.plot(*util.swap(*processed.gethist('f/Af','minibatchnumber')),'bo-',label='||f||/||Af||')
	ax.legend()
	ax.set_yscale('log')
	ax.grid(True,which='major',ls='-',axis='y')
	ax.grid(True,which='minor',ls=':',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'ratio.pdf'),fig=fig)


	fig,ax=plt.subplots()
	ax.set_title('weight norms for learner '+info)

	weightnorms,minibatches=processed.gethist('learner weightnorms','minibatchnumber')
	for l,layer in enumerate(zip(*weightnorms)):
		ax.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
	ax.legend()
	ax.set_ylim(bottom=0)
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)


	fig,ax=plt.subplots()
	ax.set_title('performance '+info)
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	cfg.savefig('{}{}'.format(cfg.outpath,'performance.pdf'),fig=fig)


def plotfunctions(sections,f,figtitle,path):
	plt.close('all')
	figs=sections.plot_y_vs_f_SI(f)
	for fignum,fig in enumerate(figs):
		fig.suptitle(figtitle)
		cfg.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
