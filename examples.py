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



def getrunfn0(target,learner):

	def runfn():
		globals().update(cfg.params)

		global unprocessed,X,X_test,Y,Y_test,sections,_learner_
		_learner_=learner
		#global learner,target,unprocessed,X,X_test,Y,Y_test,sections


		unprocessed=cfg.ActiveMemory()
		try:
			cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,unprocessed),40,name='bars')
		except:
			pass

		X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
		X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

		cfg.logcurrenttask('preparing training data')
		Y=target.eval(X)
		cfg.logcurrenttask('preparing test data')
		Y_test=target.eval(X_test)

		trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize,lossfn=util.SI_loss) #,lossgrad=mv.gen_lossgrad(AS,lossfn=util.SI_loss))

		sc1=cfg.Scheduler(cfg.nonsparsesched(iterations,start=100))
		sc2=cfg.Scheduler(cfg.sparsesched(iterations,start=1000))
		lazyplot=cfg.Clockedworker()

		cfg.logcurrenttask('preparing slices for plotting')
		sections=pt.genCrossSections(X,Y,target.eval)

		cfg.logcurrenttask('begin training')
		for i in range(iterations+1):

			cfg.poke()
			loss=trainer.step()

			unprocessed.addcontext('minibatchnumber',i)
			unprocessed.remember('minibatch loss',loss)

			if sc1.activate(i):
				unprocessed.remember('weights',learner.weights)

			if sc2.activate(i):
				fplot()
				lazyplot.do_if_rested(.2,lplot)

	return runfn



def lplot():
	processandplot(unprocessed,_learner_,X_test,Y_test)
def fplot():
	figtitle='target {}, learner {}-{}'.format(targettype,learneractivation,learnertype)
	figpath='{}target={} learner={}-{} {} minibatches'.format(cfg.outpath,targettype,learneractivation,learnertype,int(unprocessed.getval('minibatchnumber')))
	plotfunctions(sections,_learner_.eval,figtitle,figpath)

def process_input(c):
	if c==108: lplot()
	if c==102: fplot()




# learning plots
####################################################################################################


def process_snapshot_0(processed,dynfunc,X,Y,i):
	processed.addcontext('minibatchnumber',i)

	f,weights=dynfunc.eval,dynfunc.weights
	processed.remember('Af norm',jnp.average(f(X[:100])**2))
	processed.remember('test loss',util.SI_loss(f(X),Y))

def plotexample_0(unprocessed,processed,info=''):
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


process_snapshot=process_snapshot_0
plotexample=plotexample_0



# function plots
####################################################################################################

class ClonedFunc(functions.Func):
	def __init__(self,pf,weights):
		self.fdescr=pf.fdescr
		self.f=pf.f
		self.weights=weights



def processandplot(unprocessed,pfunc,X,Y,process_snapshot_fn=None,plotexample_fn=None):

	if process_snapshot_fn==None: process_snapshot_fn=process_snapshot
	if plotexample_fn==None: plotexample_fn=plotexample

	processed=cfg.ActiveMemory()

	weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')
	for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):

		cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))
		process_snapshot(processed,ClonedFunc(pfunc,weights),X,Y,i)		

	plotexample(unprocessed,processed)
	cfg.save(processed,cfg.outpath+'data')

	cfg.clearcurrenttask()
	return processed

def plotfunctions(sections,f,figtitle,path):
	cfg.logcurrenttask('generating function plots')
	plt.close('all')
	for fignum,section in enumerate(sections):
		fig=section.plot_y_vs_f_SI(f)
		cfg.trackcurrenttask('generating functions plots',(fignum+1)/len(sections))
		fig.suptitle(figtitle)
		cfg.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
	cfg.clearcurrenttask()




# prep
####################################################################################################


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


def pickdisplay():
	try:
		return cfg.displaymode
	except:
		try:
			return cfg.selectonefromargs('nodisplay','logdisplay','display0')
		except:
			return 'fulldisplay'
	

def runexample(runfn):
	displaymode=pickdisplay()

	db.clear()
	if displaymode=='fulldisplay':
		import run_in_display
		run_in_display.RID(runfn,process_input)
	else:
		if displaymode=='logdisplay':
			class LogDisplay:
				def poke(self,*args):
					if args==('log',):
						print(session.getcurrentval('log'))
			logdisp=LogDisplay()
			session.addlistener(logdisp)
		elif displaymode=='display0':
			cfg.dashboard=db.Dashboard0()
		elif displaymode=='nodisplay':
			pass
		else:
			raise ValueError
		runfn()


def runexample0(target,learner):
	runexample(getrunfn0(target,learner))
