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


def adjustparams():
	try:
		learneractivation={'t':'tanh','s':'softplus','r':'relu','lr':'leakyrelu'}[cfg.selectone({'t','s','r','lr'},cfg.cmdparams)]
		pass
	except:
		db.clear()
		print(instructions)
		quit()
	examples.adjustparams(learneractivation=learneractivation)



def run():
	globals().update(cfg.params)

	global learner,target,unprocessed,X,X_test,Y,Y_test,sections


	unprocessed=cfg.ActiveMemory()
	try:
		cfg.dashboard.add_display(examples.Display2(10,cfg.dashboard.width,unprocessed),40,name='bars')
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
			fplot()
			lazyplot.do_if_rested(.2,lplot)










def process_snapshot(processed,dynfunc,X,Y,i):
	examples.process_snapshot(processed,dynfunc,X,Y,i)
	processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in dynfunc.weights[0]]))

def plotexample(unprocessed,processed,info=''):
	examples.plotexample(unprocessed,processed,info)

	fig,ax=plt.subplots()
	ax.set_title('weight norms for learner '+info)

	weightnorms,minibatches=processed.gethist('learner weightnorms','minibatchnumber')
	for l,layer in enumerate(zip(*weightnorms)):
		ax.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
	ax.legend()
	ax.set_ylim(bottom=0)
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)






	


def lplot():
	examples.processandplot(unprocessed,functions.ParameterizedFunc(learner),X_test,Y_test,process_snapshot_fn=process_snapshot,plotexample_fn=plotexample)
def fplot():
	figtitle='target activation {}, learner activation {}'.format(targetactivation,learneractivation)
	figpath='{}target={} learner={} {} minibatches'.format(cfg.outpath,targetactivation,learneractivation,int(unprocessed.getval('minibatchnumber')))
	examples.plotfunctions(sections,learner.eval,figtitle,figpath)

def process_input(c):
	if c==108: lplot()
	if c==102: fplot()



if __name__=='__main__':
	adjustparams()

	examples.runexample(run,process_input)
