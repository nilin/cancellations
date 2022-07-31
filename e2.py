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
import math
import dashboard as db
import time
import testing
import AS_tools
import AS_HEAVY
import examplefunctions
import AS_functions as ASf
import e1

#jax.config.update("jax_enable_x64", True)



# e2
explanation='Example 2\n\
In order not to give an unfair advantage to either activation function \n\
we let the target function in this example be the sum of two antisymmetrized NNs, \n\
one constructed with each activation function. Both NNs are normalized to have the same magnitude.'


exname='e2'


def run(cmdargs):

	params={
	'targettype':'AS_NN',
	'learnertype':'AS_NN',
	'n':5,
	'd':1,
	'samples_train':10000,
	'samples_test':1000,
	'samples_quicktest':100,
	'targetwidths':[5,100,1],
	'learnerwidths':[5,100,1],
	#'targetactivation':'tanh',
	#'learneractivation':'ReLU',
	'checkpoint_interval':5,
	'timebound':cfg.hour
	}
	args,redefs=cfg.parse_cmdln_args(cmdargs)



	# e2
	try:
		l_a={'r':'ReLU','relu':'ReLU','ReLU':'ReLU','t':'tanh','tanh':'tanh'}[args[0]]
	except:
		print(10*'\n'+'Pass activation function as first parameter.\n'+db.wideline()+10*'\n')	
		sys.exit(0)

	params['learneractivation']=l_a

	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)


	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','d','checkpoint_interval'}

	cfg.outpaths.add('outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID))
	sessioninfo=explanation+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------
	cfg.log('Generating AS functions.')

	targets=[ASf.init_target(targettype,n,d,targetwidths,ac) for ac in ['ReLU','tanh']]

	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,learneractivation)

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	# e2
	cfg.log('normalizing target terms')
	targets=[util.normalize(target,X[:100]) for target in targets]
	target=jax.jit(lambda X:targets[0](X)+targets[1](X))
	target=AS_HEAVY.makeblockwise(target)

	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner.as_static(),n,d)

	cfg.log('Generating data Y.')
	Y=target(X)
	Y_test=target(X_test)



	#
	sections=pt.CrossSections(X,Y,target,3)	

	cfg.register(locals()|globals(),'learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation')
	plotter=e1.DynamicPlotter(locals()|globals(),['X_test','Y_test','learnerinitparams','learneractivation','sections'],['minibatch loss'])

	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	


	trainer=learning.Trainer(learner,X,Y)
	sc0=cfg.Scheduler(cfg.stepwiseperiodicsched([1,10],[0,120,timebound]))
	sc1=cfg.Scheduler(cfg.stepwiseperiodicsched([60],[0,timebound]))
	sc2=cfg.Scheduler(cfg.stepwiseperiodicsched([10],[0,timebound]))
	sc3=cfg.Scheduler(cfg.expsched(5,timebound,1))
	sc4=cfg.Scheduler(cfg.expsched(15,timebound,1))
	#sc2=cfg.Scheduler(cfg.stepwiseperiodicsched([2],[0,timebound])) # for testing
	#sc3=cfg.Scheduler(cfg.stepwiseperiodicsched([2],[0,timebound])) # for testing
	cfg.log('\nStart training.\n')



	while True:
		trainer.step()
		cfg.pokelisteners('refresh')

		if sc0.dispatch():
			trainer.checkpoint()

		if sc1.dispatch():
			trainer.save()

		if sc2.dispatch():
			cfg.trackcurrent('quick test loss',e1.quicktest(learner,X_test,Y_test,samples_quicktest))

		if sc3.dispatch():
			fig1=e1.getfnplot(sections,learner.as_static())
			cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

		if sc4.dispatch():
			plotter.process_state(learner)
			plotter.plotlosshist()
			plotter.plotweightnorms()
			plotter.plot3()
			plt.close('all')

		


#----------------------------------------------------------------------------------------------------


class CompPlotter():
	def __init__(self,datapaths):
		self.plotters={ac:e1.LoadedPlotter(datapaths[ac]) for ac in activations}

	def prep(self,schedule):
		for ac,plotter in self.plotters.items():
			plotter.filtersnapshots(schedule)
			plotter.prep()

	def compareweightnorms(self):
		fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,7))

		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
		rw1norms,rw2norms=zip(*rtslices)
		tw1norms,tw2norms=zip(*ttslices)

		ax1.set_title('layer 1')
		ax1.plot(rts,rw1norms,'bo-',label='ReLU',markersize=2,lw=1)
		ax1.plot(tts,tw1norms,'rd--',label='tanh',markersize=2,lw=1)

		ax2.set_title('layer 2')
		ax2.plot(rts,rw2norms,'bo-',label='ReLU',markersize=2,lw=1)
		ax2.plot(tts,tw2norms,'rd--',label='tanh',markersize=2,lw=1)

		ax1.legend()	
		ax2.legend()	
		cfg.savefig_('weightcomp.pdf',fig=fig)


	def comp3(self):
		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
		_,rfnorm=self.plotters['ReLU'].gethist('NS norm')
		_,tfnorm=self.plotters['tanh'].gethist('NS norm')
		_,rlosses=self.plotters['ReLU'].gethist('test loss')
		_,tlosses=self.plotters['tanh'].gethist('test loss')
		rweightnorms=[np.sqrt(x**2+y**2) for x,y in rtslices]
		tweightnorms=[np.sqrt(x**2+y**2) for x,y in ttslices]


		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

		ax1.plot(rweightnorms,jnp.sqrt(jnp.array(rlosses)),'bo-',markersize=2,lw=1,label='ReLU')
		ax1.plot(tweightnorms,jnp.sqrt(jnp.array(tlosses)),'rd:',markersize=2,lw=1,label='tanh')
		ax1.set_xlabel('weights')
		ax1.set_ylabel('l2 loss')
		ax1.annotate('start',(rweightnorms[0],jnp.sqrt(rlosses[0])))
		ax1.annotate('end',(rweightnorms[-1],jnp.sqrt(rlosses[-1])))
		ax1.annotate('start',(tweightnorms[0],jnp.sqrt(tlosses[0])))
		ax1.annotate('end',(tweightnorms[-1],jnp.sqrt(tlosses[-1])))

		ax2.plot(rfnorm,jnp.sqrt(jnp.array(rlosses)),'bo-',markersize=2,lw=1,label='ReLU')
		ax2.plot(tfnorm,jnp.sqrt(jnp.array(tlosses)),'rd:',markersize=2,lw=1,label='tanh')
		ax2.set_xlabel('||f||')
		ax2.set_ylabel('l2 loss')
		ax2.annotate('start',(rfnorm[0],jnp.sqrt(rlosses[0])))
		ax2.annotate('end',(rfnorm[-1],jnp.sqrt(rlosses[-1])))
		ax2.annotate('start',(tfnorm[0],jnp.sqrt(tlosses[0])))
		ax2.annotate('end',(tfnorm[-1],jnp.sqrt(tlosses[-1])))

		ax3.plot(rweightnorms,rfnorm,'bo-',markersize=2,lw=1,label='ReLU')
		ax3.plot(tweightnorms,tfnorm,'rd:',markersize=2,lw=1,label='tanh')
		ax3.set_xlabel('weights')
		ax3.set_ylabel('||f||')
		ax3.annotate('start',(rweightnorms[0],rfnorm[0]))
		ax3.annotate('end',(rweightnorms[-1],rfnorm[-1]))
		ax3.annotate('start',(tweightnorms[0],tfnorm[0]))
		ax3.annotate('end',(tweightnorms[-1],tfnorm[-1]))

		ax1.legend()	
		ax2.legend()	
		ax3.legend()	
		cfg.savefig_('comp3.pdf',fig=fig)
#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1()

	cfg.trackduration=True
	run(sys.argv[1:])
