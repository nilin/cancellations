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

#jax.config.update("jax_enable_x64", True)





cfg.outpaths.add('outputs/e1/{}/'.format(cfg.sessionID))


explanation='Example 1\n'



def run(cmdargs):

	params={
	'targettype':'AS_NN',
	'learnertype':'AS_NN',
	'n':5,
	'd':1,
	'samples_train':10000,
	'samples_test':10000,
	'samples_quicktest':100,
	#'targetwidths':[5,25,25,1],
	'targetwidths':[5,100,1],
	'learnerwidths':[5,100,1],
	'targetactivation':'tanh',
	'learneractivation':'ReLU',
	'checkpoint_interval':5,
	'timebound':60
	}


	_,redefs=cfg.parse_cmdln_args(cmdargs)
	if 'n' in redefs:
		params['targetwidths'][0]=redefs['n']
		params['learnerwidths'][0]=redefs['n']

	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)


	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','d','checkpoint_interval'}
	if 'NN' not in targettype: ignore.update({'targetwidths','targetactivation'})

	sessioninfo=explanation+'\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.write(sessioninfo,*[path+'info.txt' for path in cfg.outpaths],mode='w')


	#----------------------------------------------------------------------------------------------------
	cfg.log('Generating AS functions.')

	targetinitparams=(n,d,targetwidths,targetactivation) if 'NN' in targettype else (n,)
	learnerinitparams=(learnertype,n,d,learnerwidths,learneractivation)
	target=ASf.init_target(targettype,*targetinitparams) 
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,learneractivation)

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	cfg.log('normalizing target')
	target=util.normalize(target,X[:100])
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
	plotter=DynamicPlotter(locals()|globals(),['X_test','Y_test','sections','learnerinitparams','learneractivation'],['minibatch loss'])
	cfg.register(locals()|globals(),'learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation')
	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	


	trainer=learning.Trainer(learner,X,Y)
	sc0=cfg.Scheduler(cfg.expsched(.1,timebound))
	sc1=cfg.Scheduler(cfg.periodicsched(5,timebound))
	sc2=cfg.Scheduler(cfg.periodicsched(10,timebound))
	cfg.log('\nStart training.\n')


	while True:
		trainer.step()
		cfg.pokelisteners('refresh')

		if sc0.dispatch():
			trainer.checkpoint()

		if sc1.dispatch():
			cfg.trackcurrent('quick test loss',quicktest(learner,X_test,Y_test,samples_quicktest))

		if sc2.dispatch():
			trainer.save()

			fig1=getfnplot(sections,learner.as_static())
			cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

			plotter.process_state(learner)
			plotter.plotlosshist()
			plotter.plotweightnorms()
			plt.close('all')

		

def quicktest(learner,X_test,Y_test,samples):
	I=np.random.choice(X_test.shape[0],samples)
	return cfg.lossfn(learner.as_static()(X_test[I]),Y_test[I])


#----------------------------------------------------------------------------------------------------

def getfnplot(sections,learner):
	fig,axs=plt.subplots(1,3,figsize=(16,4))
	sections.plot(axs,learner)
	return fig



class Plotter(pt.Plotter):

	def process_state(self,learner,t=None):
		if t==None:t=cfg.timestamp()
		X_test=self.static['X_test'][:1000]
		Y_test=self.static['Y_test'][:1000]
		self.static['learneractivation']=self.static['learnerinitparams'][-1]

		self.remember('test loss',cfg.lossfn(learner.as_static()(X_test),Y_test),t)
		self.remember('NS norm',util.norm(learner.static_NS()(X_test)),t)
		self.remember('AS norm',util.norm(learner.as_static()(X_test)),t)
		self.remember('weight norms',[util.norm(W) for W in learner.weights[0]],t)

	
	def plotlosshist(self):
		fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,7))
		fig.suptitle(self.static['learneractivation'])

		def plotax(ax):
			ax.scatter(*self.gethist('minibatch loss'),color='r',label='training loss',s=.3,alpha=.3)
			ax.plot(*self.gethist('test loss'),'bo-',label='test loss',markersize=3,lw=1)
			ax.legend()
			ax.set_xlabel('seconds')

		plotax(ax1)
		plotax(ax2)
		ax1.set_ylim(0,1)
		ax2.set_yscale('log')
		cfg.savefig_('losses.pdf',fig=fig)

	def plotweightnorms(self):
		fig,ax=plt.subplots()
		fig.suptitle(self.static['learneractivation'])

		ts,tslices=self.gethist('weight norms')
		w1norms,w2norms=zip(*tslices)
		ax.plot(ts,w1norms,'bo-',label='layer 1 weights',markersize=2,lw=1)
		ax.plot(ts,w2norms,'rd--',label='layer 2 weights',markersize=2,lw=1)
		ax.legend()	
		cfg.savefig_('weightnorms.pdf',fig=fig)

	def plot3(self):
		ts,tslices=self.gethist('weight norms')
		_,fnorm=self.gethist('NS norm')
		_,losses=self.gethist('test loss')
		weightnorms=[np.sqrt(x**2+y**2) for x,y in tslices]


		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
		fig.suptitle(self.static['learneractivation'])

		ax1.plot(weightnorms,jnp.sqrt(jnp.array(losses)),'bo-',markersize=2,lw=1)
		ax1.set_xlabel('weights')
		ax1.set_ylabel('l2 loss')
		ax1.annotate('start',(weightnorms[0],jnp.sqrt(losses[0])))
		ax1.annotate('end',(weightnorms[-1],jnp.sqrt(losses[-1])))

		ax2.plot(fnorm,jnp.sqrt(jnp.array(losses)),'bo-',markersize=2,lw=1)
		ax2.set_xlabel('||f||')
		ax2.set_ylabel('l2 loss')
		ax2.annotate('start',(fnorm[0],jnp.sqrt(losses[0])))
		ax2.annotate('end',(fnorm[-1],jnp.sqrt(losses[-1])))

		ax3.plot(weightnorms,fnorm,'bo-',markersize=2,lw=1)
		ax3.set_xlabel('weights')
		ax3.set_ylabel('||f||')
		ax3.annotate('start',(weightnorms[0],fnorm[0]))
		ax3.annotate('end',(weightnorms[-1],fnorm[-1]))

		cfg.savefig_('plot3.pdf',fig=fig)

	def plotfn(self,staticlearner,figname='fnplot'):
		fig=getfnplot(self.static['sections'],staticlearner)
		cfg.savefig_(figname+'.pdf',fig=fig)



class DynamicPlotter(Plotter):

	def __init__(self,lcls,statics,trackedvars):
		super().__init__()
		for name in statics:
			#self.static[name]=cfg.getval(name)
			self.static[name]=lcls[name]
		for name in trackedvars:
			self.hists[name]=cfg.sessionstate.linkentry(name)


class LoadedPlotter(cfg.LoadedState,Plotter):

	def __init__(self,path):
		super().__init__(path)
		self.loadlearnerclone()
		
	
"""
#	def getnormplots(plotdata):
#		fig1,(ax11,ax12)=plt.subplots(1,2,figsize=(15,7))
#		fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))
#
#		def plotnormhist(ax1,ax2,plotdata):
#			NSnorm=plotdata['NS norm']
#			ASnorm=plotdata['AS norm']
#
#			ts,NSnorm,ASnorm=zip(*zip(NSnorm['timestamps'],NSnorm['vals'],ASnorm['vals']))
#			Af_over_f=jnp.array(ASnorm)/jnp.array(NSnorm)
#
#			ax1.plot(ts,NSnorm,'rd--',label='||f||')
#			ax1.plot(ts,1/Af_over_f,'bo-',label='||f||/||Af||')
#
#			ax2.plot(ts,ASnorm,'rd--',label='||Af||')
#			ax2.plot(ts,Af_over_f,'bo-',label='||Af||/||f||')
#			
#			for ax in [ax1,ax2]:
#				ax.legend()
#				ax.set_xlabel('seconds')
#				ax.grid(which='both')
#
#		plotnormhist(ax11,ax21,plotdata) # f/A,A/f
#		plotnormhist(ax12,ax22,plotdata) # f/A,A/f log plots
#		ax12.set_yscale('log')
#		ax22.set_yscale('log')
#		return fig1,fig2
#
#
#
#
#
#
#def getnormplots(plotdata):
#	plt.close('all')
#	fig1,(ax11,ax12)=plt.subplots(1,2,figsize=(15,7))
#	fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))
#
#	def plotnormhist(ax1,ax2,plotdata):
#		NSnorm=plotdata['NS norm']
#		ASnorm=plotdata['AS norm']
#
#		ts,NSnorm,ASnorm=zip(*zip(NSnorm['timestamps'],NSnorm['vals'],ASnorm['vals']))
#		Af_over_f=jnp.array(ASnorm)/jnp.array(NSnorm)
#
#		ax1.plot(ts,NSnorm,'rd--',label='||f||')
#		ax1.plot(ts,1/Af_over_f,'bo-',label='||f||/||Af||')
#
#		ax2.plot(ts,ASnorm,'rd--',label='||Af||')
#		ax2.plot(ts,Af_over_f,'bo-',label='||Af||/||f||')
#		
#		for ax in [ax1,ax2]:
#			ax.legend()
#			ax.set_xlabel('seconds')
#			ax.grid(which='both')
#
#	plotnormhist(ax11,ax21,plotdata) # f/A,A/f
#	plotnormhist(ax12,ax22,plotdata) # f/A,A/f log plots
#	ax12.set_yscale('log')
#	ax22.set_yscale('log')
#	return fig1,fig2
#
#
"""

#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1()

	cfg.trackduration=True
	run(sys.argv[1:])
