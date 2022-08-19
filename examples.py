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




def processandplot(unprocessed,pfunc,X,Y,process_snapshot_fn=None,plotexample_fn=None):

	if process_snapshot_fn==None: process_snapshot_fn=process_snapshot
	if plotexample_fn==None: plotexample_fn=plotexample

	processed=cfg.ActiveMemory()

	weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')
	for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):

		cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))
		process_snapshot(processed,functions.DynFunc(pfunc,weights),X,Y,i)		

	plotexample(unprocessed,processed)
	cfg.save(processed,cfg.outpath+'data')

	cfg.clearcurrenttask()
	return processed




def process_snapshot(processed,dynfunc,X,Y,i):
	processed.addcontext('minibatchnumber',i)

	f,weights=dynfunc.eval,dynfunc.weights
	processed.remember('Af norm',jnp.average(f(X[:100])**2))
	processed.remember('test loss',util.SI_loss(f(X),Y))



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
	ax.set_title('performance '+info)
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	cfg.savefig('{}{}'.format(cfg.outpath,'performance.pdf'),fig=fig)







def plotfunctions(sections,f,figtitle,path):
	cfg.logcurrenttask('generating function plots')
	plt.close('all')
	for fignum,section in enumerate(sections):
		fig=section.plot_y_vs_f_SI(f)
		cfg.trackcurrenttask('generating functions plots',(fignum+1)/len(sections))
		fig.suptitle(figtitle)
		cfg.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
	cfg.clearcurrenttask()




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



class Display2(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
		self.addbar('minibatch loss',style='.')
		self.addbar('minibatch loss',style=db.BOX,avg_of=100)
		self.addspace()
		self.addline()
		self.addnumberprint('minibatchnumber',msg='minibatch number {:.0f}')



def runexample(run,process_input):
	db.clear()
	if 'nodisplay' in cfg.cmdparams:
		run()
	elif 'logdisplay' in cfg.cmdparams:
		class LogDisplay:
			def poke(self,*args):
				if args==('log',):
					print(session.getcurrentval('log'))
		logdisp=LogDisplay()
		session.addlistener(logdisp)
		run()
	elif 'display0' in cfg.cmdparams:
		cfg.dashboard=db.Dashboard0()
		run()
	else:
		import run_in_display
		run_in_display.RID(run,process_input)
