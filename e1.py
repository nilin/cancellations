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
#'learneractivation':'ReLU',
'checkpoint_interval':5,
'timebound':600,
'fnplotfineness':250
}


def run():

	try:
		l_a={'r':'ReLU','relu':'ReLU','ReLU':'ReLU','t':'tanh','tanh':'tanh'}[cfg.cmdparams[0]]
	except:
		print(10*'\n'+'Pass activation function as first parameter.\n'+db.wideline()+10*'\n')	
		sys.exit(0)

	params['learneractivation']=l_a
	if 'n' in cfg.cmdredefs:
		params['targetwidths'][0]=cfg.cmdredefs['n']
		params['learnerwidths'][0]=cfg.cmdredefs['n']

	globals().update(params)
	globals().update(cfg.cmdredefs)
	varnames=cfg.orderedunion(params,cfg.cmdredefs)


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
	sections=pt.CrossSections(X,Y,target,3,fineness=fnplotfineness)	

	reg_args=['learnerinitparams','X','Y','X_test','Y_test','sections','learneractivation']
	cfg.register(locals()|globals(),*reg_args)
	dynamicplotter=DynamicPlotter(locals()|globals(),reg_args,['minibatch loss','weights'])


	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------

	trainer=learning.Trainer(learner,X,Y)
	sc0=cfg.Scheduler(cfg.stepwiseperiodicsched([1,10],[0,120,timebound]))
	sc1=cfg.Scheduler(cfg.stepwiseperiodicsched([60],[0,timebound]))
	sc2=cfg.Scheduler(cfg.stepwiseperiodicsched([10],[0,timebound]))
	sc3=cfg.Scheduler(cfg.expsched(5,timebound,.2))
	sc4=cfg.Scheduler(cfg.stepwiseperiodicsched([5,30],[0,120,timebound]))
	cfg.log('\nStart training.\n')


	while True:
		try:
			trainer.step()
			cfg.pokelisteners('refresh')

			if sc0.dispatch():
				trainer.checkpoint()

			if sc1.dispatch():
				trainer.save()

			if sc2.dispatch():
				cfg.trackcurrent('quick test loss',quicktest(learner,X_test,Y_test,samples_quicktest))

			if sc3.dispatch():
				"""
				fig1=getfnplot(sections,learner.as_static())
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
				"""
				pass

			if sc4.dispatch():
				"""
				dynamicplotter.process_state(learner)
				dynamicplotter.learningplots()
				"""
				pass

		except KeyboardInterrupt:
			db.clear()			
			inp=input('Enter to continute, p+Enter to plot, q+Enter to end.\n')
			if inp=='p':
				fig1=getfnplot(sections,learner.as_static())
				cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)

				temp_plotter=DynamicPlotter(locals()|globals(),reg_args,['minibatch loss','weights'])
				temp_plotter.prep(sc3.schedule)
				temp_plotter.learningplots()
				del temp_plotter
			if inp=='q':
				break
			db.clear()			

		

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
		wnorms=zip(*tslices)
		
		#styles=['bo-','r0-','bd:','rd:']
		colors=['r','g','b']
		for i,wnorm in enumerate(wnorms):
			ax.plot(ts,wnorm,'o-',color=colors[i],label='layer {} weights'.format(i+1),markersize=2,lw=1)
		ax.legend()	
		cfg.savefig_('weightnorms.pdf',fig=fig)

	def plot3(self):
		ts,tslices=self.gethist('weight norms')
		_,fnorm=self.gethist('NS norm')
		_,losses=self.gethist('test loss')
		weightnorms=[max([weights for weights in lweights]) for lweights in tslices]


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
		plt.close('all')

	def learningplots(self):
		self.plotlosshist()
		self.plotweightnorms()
		self.plot3()
		plt.close('all')


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




class CompPlotter():
	def __init__(self,datapaths):
		self.plotters={ac:LoadedPlotter(datapaths[ac]) for ac in activations}

	def prep(self,schedule):
		for ac,plotter in self.plotters.items():
			plotter.filtersnapshots(schedule)
			plotter.prep()

	def compareweightnorms(self):
		fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,7))

		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
		rw1norms,rw2norms,*_=zip(*rtslices)
		tw1norms,tw2norms,*_=zip(*ttslices)

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
		rweightnorms=[np.sqrt(x**2+y**2) for x,y,*_ in rtslices]
		tweightnorms=[np.sqrt(x**2+y**2) for x,y,*_ in ttslices]


		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

		ax1.plot(rweightnorms,jnp.sqrt(jnp.array(rlosses)),'bo-',markersize=2,lw=1,label='ReLU')
		ax1.plot(tweightnorms,jnp.sqrt(jnp.array(tlosses)),'rd:',markersize=2,lw=1,label='tanh')
		ax1.set_xlabel('weights')
		ax1.set_ylabel('l2 loss')
		ax1.annotate('start',(rweightnorms[0],jnp.sqrt(rlosses[0])))
		ax1.annotate('end',(rweightnorms[-1],jnp.sqrt(rlosses[-1])))
		ax1.annotate('start',(tweightnorms[0],jnp.sqrt(tlosses[0])))


#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1(params)

	cfg.trackduration=True
	run()
