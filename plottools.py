import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
import optax
import math
import numpy as np
import sys
import AS_functions
import matplotlib.pyplot as plt
import numpy as np
import learning
import pdb
from collections import deque




def samplepoints(X,Y,nsamples):
	p=Y**2
	p=p/jnp.sum(p)
	#I=jnp.random.choice(,range(len(p)),nsamples,p=p)
	I=rnd.choice(cfg.nextkey(),jnp.arange(len(p)),(nsamples,),p=p)
	return X[I]
	



def linethrough(x,interval):
	corner=np.zeros_like(x)
	corner[0][0]=1
	x_rest=(1-corner)*x
	X=interval[:,None,None]*corner[None,:,:]+x_rest[None,:,:]
	return X




class CrossSections:
	def __init__(self,X,Y,target,nsections,fineness=500):

		cfg.log('Preparing cross sections for plotting.')
		x0s=samplepoints(X,Y,nsections)
		self.interval=jnp.arange(-1,1,2/fineness)
		self.lines=[linethrough(x0,self.interval) for x0 in x0s]
		self.ys=[target(line) for line in self.lines]

	def plot(self,axs,learned):
		for ax,x,y in zip(axs,self.lines,self.ys):
			ax.plot(self.interval,y,'b',label='target')
			ax.plot(self.interval,learned(x),'r',label='learned')
			ax.legend()
		



class AbstractPlotter(cfg.State):

	def filtersnapshots(self,schedule):

		#self.weightshist['timestamps'],self.weightshist['vals']=cfg.filterschedule(schedule,self.weightshist['timestamps'],self.weightshist['vals'])
		self.hists['weights']['timestamps'],self.hists['weights']['vals']=cfg.filterschedule(schedule,self.hists['weights']['timestamps'],self.hists['weights']['vals'])

	def loadlearnerclone(self):
		self.emptylearner=AS_functions.gen_learner(*self.static['learnerinitparams'])

	def getlearner(self,weights):
		return self.emptylearner.reset(weights)

	def getstaticlearner(self,weights):
		return self.getlearner(weights).as_static()

	def prep(self,schedule=None):

		if schedule!=None:
			self.filtersnapshots(schedule)

		self.loadlearnerclone()
		timestamps,states=self.hists['weights']['timestamps'],self.hists['weights']['vals']
		for i,(t,state) in enumerate(zip(timestamps,states)):

			cfg.log('processing snapshot {}/{}'.format(i+1,len(timestamps)))
			self.process_state(self.getlearner(state),t)




def getfnplot(sections,learner):
	fig,axs=plt.subplots(1,3,figsize=(16,4))
	sections.plot(axs,learner)
	return fig



class Plotter(AbstractPlotter):

	def process_state(self,learner,t=None):

		cfg.dbprint('process_state called')

		if t==None:t=self.timestamp()
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
		#self.plot3()
		plt.close('all')


class DynamicPlotter(Plotter):

	def __init__(self,lcls,statics,trackedvars):
		super().__init__()
		for name in statics:
			#self.static[name]=cfg.getval(name)
			self.static[name]=lcls[name]

		for name,hist in trackedvars.items():
			self.hists[name]=hist #cfg.sessionstate.linkentry(name)

		#cfg.dblog('dp')
		#self.weightshist=self.static['weightshistpointer']		


class LoadedPlotter(cfg.LoadedState,Plotter):

	def __init__(self,path):
		super().__init__(path)
		self.loadlearnerclone()
		#self.weightshist=self.static['weightshistpointer']



class CompPlotter():
	def __init__(self,datapaths):
		self.plotters={ac:LoadedPlotter(datapaths[ac]) for ac in ['ReLU','tanh']}

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


#	def plot3(self):
#		ts,tslices=self.gethist('weight norms')
#		_,fnorm=self.gethist('NS norm')
#		_,losses=self.gethist('test loss')
#		weightnorms=[max([weights for weights in lweights]) for lweights in tslices]
#
#
#		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
#		fig.suptitle(self.static['learneractivation'])
#
#		ax1.plot(weightnorms,jnp.sqrt(jnp.array(losses)),'bo-',markersize=2,lw=1)
#		ax1.set_xlabel('weights')
#		ax1.set_ylabel('l2 loss')
#		ax1.annotate('start',(weightnorms[0],jnp.sqrt(losses[0])))
#		ax1.annotate('end',(weightnorms[-1],jnp.sqrt(losses[-1])))
#
#		ax2.plot(fnorm,jnp.sqrt(jnp.array(losses)),'bo-',markersize=2,lw=1)
#		ax2.set_xlabel('||f||')
#		ax2.set_ylabel('l2 loss')
#		ax2.annotate('start',(fnorm[0],jnp.sqrt(losses[0])))
#		ax2.annotate('end',(fnorm[-1],jnp.sqrt(losses[-1])))
#
#		ax3.plot(weightnorms,fnorm,'bo-',markersize=2,lw=1)
#		ax3.set_xlabel('weights')
#		ax3.set_ylabel('||f||')
#		ax3.annotate('start',(weightnorms[0],fnorm[0]))
#		ax3.annotate('end',(weightnorms[-1],fnorm[-1]))
#
#		cfg.savefig_('plot3.pdf',fig=fig)

	def plotfn(self,staticlearner,figname='fnplot'):
		fig=getfnplot(self.static['sections'],staticlearner)
		cfg.savefig_(figname+'.pdf',fig=fig)
		plt.close('all')

	def learningplots(self):
		self.plotlosshist()
		self.plotweightnorms()
		self.plot3()
		plt.close('all')


#class DynamicPlotter(Plotter):
#
#	def __init__(self,lcls,statics,trackedvars):
#		super().__init__()
#		for name in statics:
#			#self.static[name]=cfg.getval(name)
#			self.static[name]=lcls[name]
#		for name in trackedvars:
#			self.hists[name]=cfg.sessionstate.linkentry(name)
#
#
#class LoadedPlotter(cfg.LoadedState,Plotter):
#
#	def __init__(self,path):
#		super().__init__(path)
#		self.loadlearnerclone()
#
#
#
#
#class CompPlotter():
#	def __init__(self,datapaths):
#		self.plotters={ac:LoadedPlotter(datapaths[ac]) for ac in activations}
#
#	def prep(self,schedule):
#		for ac,plotter in self.plotters.items():
#			plotter.filtersnapshots(schedule)
#			plotter.prep()
#
#	def compareweightnorms(self):
#		fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,7))
#
#		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
#		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
#		rw1norms,rw2norms,*_=zip(*rtslices)
#		tw1norms,tw2norms,*_=zip(*ttslices)
#
#		ax1.set_title('layer 1')
#		ax1.plot(rts,rw1norms,'bo-',label='ReLU',markersize=2,lw=1)
#		ax1.plot(tts,tw1norms,'rd--',label='tanh',markersize=2,lw=1)
#
#		ax2.set_title('layer 2')
#		ax2.plot(rts,rw2norms,'bo-',label='ReLU',markersize=2,lw=1)
#		ax2.plot(tts,tw2norms,'rd--',label='tanh',markersize=2,lw=1)
#
#		ax1.legend()	
#		ax2.legend()	
#		cfg.savefig_('weightcomp.pdf',fig=fig)
#
#
#	def comp3(self):
#		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
#		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
#		_,rfnorm=self.plotters['ReLU'].gethist('NS norm')
#		_,tfnorm=self.plotters['tanh'].gethist('NS norm')
#		_,rlosses=self.plotters['ReLU'].gethist('test loss')
#		_,tlosses=self.plotters['tanh'].gethist('test loss')
#		rweightnorms=[np.sqrt(x**2+y**2) for x,y,*_ in rtslices]
#		tweightnorms=[np.sqrt(x**2+y**2) for x,y,*_ in ttslices]
#
#
#		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
#
#		ax1.plot(rweightnorms,jnp.sqrt(jnp.array(rlosses)),'bo-',markersize=2,lw=1,label='ReLU')
#		ax1.plot(tweightnorms,jnp.sqrt(jnp.array(tlosses)),'rd:',markersize=2,lw=1,label='tanh')
#		ax1.set_xlabel('weights')
#		ax1.set_ylabel('l2 loss')
#		ax1.annotate('start',(rweightnorms[0],jnp.sqrt(rlosses[0])))
#		ax1.annotate('end',(rweightnorms[-1],jnp.sqrt(rlosses[-1])))
#		ax1.annotate('start',(tweightnorms[0],jnp.sqrt(tlosses[0])))
#
