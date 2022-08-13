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


exname='exRatio2'

explanation='Example '+exname#+': softplus target function'
#timebound=cfg.hour

params={
'ftype':'AS_NN',
'n':5,
'd':1,
'learnerwidths':[5,250,1],
'targetactivation':'tanh',
'weight_decay':.1,
'lossfn':'SI_loss',
'samples_rademacher':100,
'timebound':600,
'minibatchsize':50
}

def setparams():

	params.update(cfg.cmdredefs)

	try:
		params['learneractivation']={'r':'ReLU','t':'tanh','d':'DReLU','p':'ptanh'}[cfg.selectone({'r','t','d','p'},cfg.cmdparams)]
	except:
		print(10*'\n'+'Pass target activation function as parameter.\n'+10*'\n')	
		raise Exception

	params['widths'][0]=params['n']


	params.update(cmdredefs)
	globals().update(params)
	cfg.varnames=list(params)

	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}
	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in cfg.varnames],separator='\n',ignore=ignore)
	session.remember('sessioninfo',sessioninfo)




def run(**kwargs):

	cfg.trackduration=True
	if len(kwargs)==0:
		cfg.print(cfg.inpath)
		datapath=cfg.longestduration(cfg.inpath)+'XY'
		cfg.print(datapath)
		data=cfg.load(datapath)

	globals().update(kwargs)


	cfg.inpath='outputs/{}/target={}/'.format(exname,targetactivation)
	cfg.outpath='outputs/{}/target={} learner={}/{}/'.format(exname,targetactivation,learneractivation,cfg.sessionID)
	cfg.write(session.getval('sessioninfo'),cfg.outpath+'info.txt',mode='w')


	cfg.lossfn=util.SI_loss
	learnerinitparams=(ftype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(*learnerinitparams)



	trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize)


	processed=cfg.ActiveMemory()
	cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,processed),40,name='bars')



	sc1=cfg.Scheduler(cfg.periodicsched(50,iterations2))
	sc2=cfg.Scheduler(cfg.stepwiseperiodicsched([10,100],[0,100,iterations2]))
	sc3=cfg.Scheduler(cfg.periodicsched(250,iterations2))


	


	for i in range(iterations2+1):

		if cfg.mode=='break':
			break
		

		try:
			loss=trainer.step()
			processed.addcontext('minibatch number',i)
			processed.remember('minibatch loss',loss)
			processed.remember('learner weightnorms',jnp.array([util.norm(l) for l in learner.weights[0]]))

			if sc1.activate(i):
				processed.remember('Af norm',jnp.average(learner.as_static()(X_test[:100])**2))
				processed.remember('f norm',jnp.average(learner.get_NS().as_static()(X_test[:100])**2))
				processed.compute(['f norm','Af norm'],lambda x,y:x/y,'f/Af')

			if sc2.activate(i):
				processed.remember('test loss',util.SI_loss(learner.as_static()(X_test),Y_test))

				cfg.save(processed,cfg.outpath+'data')

			if sc3.activate(i):
				plotexample(processed,learneractivation)
				plt.close('all')
				figs=sections.plot_y_vs_f_SI(learner.as_static())
				for fignum,fig in enumerate(figs):
					fig.suptitle('learner activation {}'.format(learneractivation))
					cfg.savefig('{}{} {} minibatches {}.pdf'.format(cfg.outpath,learneractivation,int(i),fignum),fig=fig)
					
			#processed.trackcurrent('minibatch number',i)
		except KeyboardInterrupt:
			break

	cfg.dashboard.del_display('bars')

	return processed



class Display2(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
		self.addbar('minibatch loss',style=db.dash,avg_of=10)
		self.addspace()
		self.addnumberprint('test loss',msg='test loss {:.3}')
		self.addbar('test loss')
		self.addline()
		self.addnumberprint('minibatch number',msg='minibatch number {:.0f}/'+str(iterations2))




def plotexample(memory,learneractivation):
	plt.close('all')

	fig,ax=plt.subplots()
	ax.set_title('test loss for learner {}'.format(learneractivation))	

	ax.plot(*util.swap(*memory.gethist('test loss','minibatch number')),'r-',label='test loss')
	ax.legend()
	ax.set_ylim(bottom=0,top=1)
	ax.grid(True,which='major',ls='-',axis='y')
	ax.grid(True,which='minor',ls=':',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)


	fig,ax=plt.subplots()
	ax.set_title('norm ratio (sign problem) for learner {}'.format(learneractivation))	

	ax.plot(*util.swap(*memory.gethist('f/Af','minibatch number')),'bo-',label='||f||/||Af||')
	ax.legend()
	ax.set_yscale('log')
	ax.grid(True,which='major',ls='-',axis='y')
	ax.grid(True,which='minor',ls=':',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'ratio.pdf'),fig=fig)


	fig,ax=plt.subplots()
	ax.set_title('weight norms for learner {}'.format(learneractivation))	

	weightnorms,minibatches=memory.gethist('learner weightnorms','minibatch number')
	for l,layer in enumerate(zip(*weightnorms)):
		ax.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
	ax.legend()
	ax.set_ylim(bottom=0)
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)


	


if __name__=='__main__':

	cfg.dashboard=db.Dashboard0()
	setparams()
	run()
