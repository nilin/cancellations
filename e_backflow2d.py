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






cfg.exname='backflow2d'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':2,
#'learnertype':'AS_NN',
#'learnerwidths':[10,100,100,1],
'learnerwidths_b':[[2,5,25],5],
'learnerwidths_p':[[2,5,100],5],
'learnerwidths_f':[[3,8,8],[10,10,10],2],
'learneractivation':'tanh',
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

instructions='instructions:\n\npython exBackflow2d.py (b/f/p) \n\nparameters represent:\nb=backflow+dets / f=ferminet / p=backflow_detsandsym (product of sym and det)\n'


def adjustparams():
	try:
		selection=cfg.selectone({'b','f','p'},cfg.cmdparams)
		learnertype={'b':'backflowdets','f':'ferminet','p':'backflow_detsandsym'}[selection]
		learnerwidths=cfg.params['learnerwidths_'+selection]
	except:
		db.clear()
		print(instructions)
		quit()

	examples.adjustparams(learnertype=learnertype,learnerwidths=learnerwidths)




def run():
	globals().update(cfg.params)

	global learner,target,unprocessed,X,X_test,Y,Y_test,sections


	unprocessed=cfg.ActiveMemory()
	try:
		cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,unprocessed),40,name='bars')
	except:
		pass

	
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	target=functions.DynFunc(ftype=targettype,n=n,d=d,widths=targetwidths,activation=targetactivation)
	learner=functions.DynFunc(ftype=learnertype,n=n,d=d,widths=learnerwidths,activation=learneractivation)

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
			lazyplot.do_if_rested(.2,fplot,lplot)




def process_snapshot(processed,dynfunc,X,Y,i):
	processed.addcontext('minibatchnumber',i)

	f,weights=dynfunc.eval,dynfunc.weights
	#processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in weights[0]]))
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

#	fig,ax=plt.subplots()
#	.
#	.
#	.

	fig,ax=plt.subplots()
	ax.set_title('performance '+info)
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	cfg.savefig('{}{}'.format(cfg.outpath,'performance.pdf'),fig=fig)


examples.process_snapshot=process_snapshot
examples.plotexample=plotexample
	

class Display2(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
		self.addbar('minibatch loss',style='.')
		self.addbar('minibatch loss',style=db.BOX,avg_of=100)
		self.addspace()
		self.addline()
		self.addnumberprint('minibatchnumber',msg='minibatch number {:.0f}/'+str(iterations))

def lplot():
	examples.processandplot(unprocessed,functions.ParameterizedFunc(learner),X_test,Y_test)
def fplot():
	figtitle='learner type {}'.format(learnertype)
	figpath='{}{} {} minibatches'.format(cfg.outpath,learneractivation,int(unprocessed.getval('minibatchnumber')))
	examples.plotfunctions(sections,learner.eval,figtitle,figpath)

def process_input(c):
	if c==108: lplot()
	if c==102: fplot()



if __name__=='__main__':
	adjustparams()
	examples.runexample(run,process_input)
