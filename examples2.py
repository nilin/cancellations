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


#jax.config.update("jax_enable_x64", True)






class Example:

	def __init__(self,outpath=None,explanation=''):
		self.explanation='Example '+self.exname+' '+explanation
		self.adjustparams()
		cfg.trackduration=True

		cfg.outpath='outputs/{}/{}/'.format(self.exname,cfg.sessionID) if outpath==None else outpath
		cfg.write(session.getval('sessioninfo'),cfg.outpath+'info.txt',mode='w')

		self.X=rnd.uniform(cfg.nextkey(),(self.samples_train,self.n,self.d),minval=-1,maxval=1)
		self.X_test=rnd.uniform(cfg.nextkey(),(self.samples_test,self.n,self.d),minval=-1,maxval=1)

		self.unprocessed=cfg.ActiveMemory()

		self.sc_w=cfg.Scheduler(cfg.stepwiseperiodicsched([10,100],[0,100,self.iterations]))
		self.i=0

		
	def adjustparams(self):

		params=self.params
		params.update(cfg.cmdredefs)

		try:
			params['learneractivation']={'r':'ReLU','t':'tanh','d':'DReLU','p':'ptanh'}[cfg.selectone({'r','t','d','p'},cfg.cmdparams)]
		except:
			print(10*'\n'+'Pass target activation function as parameter.\n'+10*'\n')	
			raise Exception

		params['learnerwidths'][0]=params['n']

		info='\n'.join(['{}={}'.format(k,v) for k,v in params.items()])
		sessioninfo='{}\nsessionID: {}\n{}'.format(self.explanation,cfg.sessionID,info)
		session.remember('sessioninfo',sessioninfo)

		#for k,v in params.items():
		#	setattr(self,k,v)
		self.__dict__.update(params)
		#print(self.__dict__)


	def run(self):
		while self.i<=self.iterations:
			self.step()


	def step(self):
		cfg.poke()
		loss=trainer.step()
		self.unprocessed.addcontext('minibatchnumber',self.i)
		self.unprocessed.remember('minibatch loss',loss)

		if self.sc_w.activate(self.i):
			self.unprocessed.remember('weights',learner.weights)


	def process_snapshot(processed,weights,i):
		processed.addcontext('minibatchnumber',i)

		AS=util.fixparams(self.AS,weights)
		NS=util.fixparams(self.NS,weights)
		X=self.X_test
		Y=self.Y_test

		processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in weights[0]]))
		processed.remember('Af norm',jnp.average(AS(X[:100])**2))
		processed.remember('f norm',jnp.average(NS(X[:100])**2))
		processed.compute(['f norm','Af norm'],lambda x,y:x/y,'f/Af')
		processed.remember('test loss',util.SI_loss(AS(X),Y))

		del AS,NS


	def processandplot(unprocessed):
		processed=cfg.ActiveMemory()

		for weights,i in zip(*unprocessed.gethist('weights','minibatchnumber')):
			self.process_snapshot(processed,weights,i)		

		self.plotexample(processed)
		cfg.save(processed,cfg.outpath+'data')
		return processed


	def plotexample(memory):

		learneractivation=session.getval('learneractivation')
		plt.close('all')

		fig,ax=plt.subplots()
		ax.set_title('test loss for learner {}'.format(learneractivation))	

		ax.plot(*util.swap(*memory.gethist('test loss','minibatchnumber')),'r-',label='test loss')
		ax.legend()
		ax.set_ylim(bottom=0,top=1)
		ax.grid(True,which='major',ls='-',axis='y')
		ax.grid(True,which='minor',ls=':',axis='y')
		cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)


		fig,ax=plt.subplots()
		ax.set_title('norm ratio (sign problem) for learner {}'.format(learneractivation))	

		ax.plot(*util.swap(*memory.gethist('f/Af','minibatchnumber')),'bo-',label='||f||/||Af||')
		ax.legend()
		ax.set_yscale('log')
		ax.grid(True,which='major',ls='-',axis='y')
		ax.grid(True,which='minor',ls=':',axis='y')
		cfg.savefig('{}{}'.format(cfg.outpath,'ratio.pdf'),fig=fig)


		fig,ax=plt.subplots()
		ax.set_title('weight norms for learner {}'.format(learneractivation))	

		weightnorms,minibatches=memory.gethist('learner weightnorms','minibatchnumber')
		for l,layer in enumerate(zip(*weightnorms)):
			ax.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
		ax.legend()
		ax.set_ylim(bottom=0)
		cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)






class ExSlater(Example):

	def __init__(self):
		exname='exSlater'

		params={
		'ftype':'AS_NN',
		'n':5,
		'd':1,
		'targettype':'HermiteSlater',
		'learnerwidths':[5,250,1],
		'weight_decay':.1,
		'lossfn':'SI_loss',
		'samples_train':100000,
		'samples_test':1000,
		'iterations':1000,
		'minibatchsize':50
		}
		cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,unprocessed),40,name='bars')

		super().__init__(exname,params)

		target=ASf.init_target(self.targettype,n)
		self.Y=target(self.X)
		self.Y_test=target(self.X_test)

		cfg.lossfn=util.SI_loss
		learnerinitparams=(ftype,n,d,learnerwidths,learneractivation)
		learner=ASf.init_learner(*learnerinitparams)
		AS,NS=learner.AS,learner.NS

		trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize)
		sections=pt.CrossSections(X,Y,target)




#	
#
#class Display2(db.StackedDisplay):
#
#	def __init__(self,height,width,memory):
#		super().__init__(height,width,memory)
#		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
#		self.addbar('minibatch loss',style=db.dash)
#		self.addbar('minibatch loss',style=db.BOX,avg_of=25)
#		self.addspace()
#		#self.addnumberprint('test loss',msg='test loss {:.3}')
#		#self.addbar('test loss')
#		self.addline()
#		self.addnumberprint('minibatchnumber',msg='minibatch number {:.0f}/'+str(iterations))
#
#
#
#
#if __name__=='__main__':
#
#	setparams()
#
#	import run_in_display
#	run_in_display.RID(run)
