import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
import optax
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import learning
import AS_HEAVY
import copy
import pdb
from config import session
from collections import deque

import warnings
warnings.filterwarnings('ignore')


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

def slicethrough(x,I):
	S,T=np.meshgrid(I,I)
	X=np.array(x)[None,None,:,:]+np.zeros_like(S)[:,:,None,None]
	X[:,:,0,0]=S
	X[:,:,0,1]=T
	return X

def slicesthrough(x,I):

	S,T=np.meshgrid(I,I)

	X=np.array(x)[None,None,:,:]+np.zeros_like(S)[:,:,None,None]
	X1=copy.deepcopy(X)
	X2=copy.deepcopy(X)
	X3=copy.deepcopy(X)

	for X_,i,j in [(X1,0,1),(X2,0,2),(X3,1,2)]:
		X_[:,:,0,i]=S
		X_[:,:,0,j]=T

	return X1,X2,X3




def genCrossSections(X,Y,target):
	cfg.log('Preparing cross sections for plotting.')
	n=X.shape[-1]
	x0s=samplepoints(X,Y,{1:3,2:3,3:1}[n])
	CrossSection=globals()['CrossSection{}D'.format(n)]
	return [CrossSection(X,Y,target,x0) for x0 in x0s]

class CrossSection:
	def __init__(self,X,Y,fineness):
		self.interval=jnp.arange(-1,1,2/fineness)
		self.X=X
		self.Y=Y
	def plot_y_vs_f_SI(self,staticlearner,normalized=True,**kwargs):
		f=util.closest_multiple(staticlearner,self.X[:250],self.Y[:250],normalized=normalized)
		return self.plot_y_vs_f(f,normalized_target=normalized,**kwargs)

class CrossSection1D(CrossSection):
	def __init__(self,X,Y,target,x0):
		super().__init__(X,Y,100)
		self.line=linethrough(x0,self.interval)
		self.y=target(self.line)

	def plot_y_vs_f(self,f,normalized_target=False):

		c=1/util.norm(self.Y) if normalized_target else 1

		fig,ax=plt.subplots()
		ax.plot(self.interval,c*self.y,'b',label='target')
		ax.plot(self.interval,f(self.line),'r',ls='dashed',label='learned')
		ax.legend()
		return fig


class CrossSection2D(CrossSection):
	def __init__(self,X,Y,target,x0):
		super().__init__(X,Y,50)
		self.slice=slicethrough(x0,self.interval)
		self.y=util.applyalonglast(target,self.slice,2)


	def plot_y_vs_f(self,f,normalized_target=False):

		I=self.interval
		c=1/util.norm(self.Y) if normalized_target else 1

		fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(17,5))
		y0=c*self.y
		y1=util.applyalonglast(f,self.slice,2)

		#M=jnp.max(jnp.abs(y0))+jnp.max(jnp.abs(y1))
		#M=jnp.max(jnp.abs(c*self.Y))
		M=1 if normalized_target else util.norm(self.Y)
		M*=4

		ax0.set_title('target')
		ax1.set_title('learner')
		ax2.set_title('both')

		im0=ax0.pcolormesh(I,I,y0,cmap='seismic',vmin=-M,vmax=M)
		im1=ax1.pcolormesh(I,I,y1,cmap='seismic',vmin=-M,vmax=M)
		im0.set_edgecolor('face')
		im1.set_edgecolor('face')

		#ax2.pcolormesh(I,I,y1-y0,cmap='seismic',vmin=-M,vmax=M)
		ax2.contour(I,I,y0,colors='b',linewidths=.5)
		ax2.contour(I,I,y1,colors='r',linewidths=.5)
		ax2.contour(I,I,y0,levels=[0],colors='b',linewidths=3)
		ax2.contour(I,I,y1,levels=[0],colors='r',linewidths=3)

		#fig.colorbar(im)
		#fig.colorbar(im1)
		#fig.colorbar(im2)
		return fig


class CrossSection3D(CrossSection):
	def __init__(self,X,Y,target,x0):
		super().__init__(X,Y,50)
		self.slices=slicesthrough(x0,self.interval)
		self.ys=[util.applyalonglast(target,sl,2) for sl in self.slices]

	def plot_y_vs_f(self,f,normalized_target=False):
		cfg.logcurrenttask('drawing plots')

		I=self.interval
		c=1/util.norm(self.Y) if normalized_target else 1

		fig,axsrows=plt.subplots(len(self.slices),3,figsize=(17,17))
		for sl,y,(ax0,ax1,ax2) in zip(self.slices,self.ys,axsrows):
			y0=c*y
			y1=util.applyalonglast(f,sl,2)
			M=jnp.max(jnp.abs(y0))
			#ax.pcolormesh(I,I,y0)
			im=ax0.pcolormesh(I,I,y1-y0,cmap='seismic',vmin=-M,vmax=M)
			ax0.contour(I,I,y0,colors='b',linewidths=.5)
			ax0.contour(I,I,y1,colors='r',linewidths=.5)
			ax0.contour(I,I,y0,levels=[0],colors='b',linewidths=3)
			ax0.contour(I,I,y1,levels=[0],colors='r',linewidths=3)

			im1=ax1.pcolormesh(I,I,y0,cmap='seismic',vmin=-M,vmax=M)
			im2=ax2.pcolormesh(I,I,y1,cmap='seismic',vmin=-M,vmax=M)

		cfg.clearcurrenttask()
		return fig



def singlefnplot_all_in_one(X,statictarget,Y=None):

	if Y==None:
		Y=statictarget(X)
	sections=CrossSections(X,Y,statictarget,3,fineness=200)	
	fig=sections.plot_y()
	del sections
	return fig

def fnplot_all_in_one(X,statictarget,staticlearner,Y=None,normalized=True):

	if Y==None:
		Y=statictarget(X)
	sections=CrossSections(X,Y,statictarget,3,fineness=200)	
	fig=sections.plot_SI(staticlearner,normalized)
	del sections
	return fig












#
#
#
#
#class Graphs:
#	def __init__(self,memory):
#		self.memory=memory
#
#	def plot(self,varname,timevar):
#		fig,ax=plt.subplots()
#
#		x,y=self.memory.gethist(varname,timevar)
#
#		ax.plot(x,y)	
#










#
#
#class AbstractPlotter(): #cfg.State):
#
#	def filtersnapshots(self,schedule):
#		self.hists['weights']['timestamps'],self.hists['weights']['vals']=cfg.filterschedule(schedule,self.hists['weights']['timestamps'],self.hists['weights']['vals'])
#
#	def loadlearnerclone(self):
#		self.emptylearner=AS_functions.gen_learner(*self.static['learnerinitparams'])
#
#	def getlearner(self,weights):
#		return self.emptylearner.reset(weights)
#
#	def getstaticlearner(self,weights):
#		return self.getlearner(weights).as_static()
#
#	def prep(self,schedule=None):
#
#		if schedule!=None:
#			self.filtersnapshots(schedule)
#
#		self.loadlearnerclone()
#		timestamps,states=self.hists['weights']['timestamps'],self.hists['weights']['vals']
#		for i,(t,state) in enumerate(zip(timestamps,states)):
#
#			cfg.log('processing snapshot {}/{}'.format(i+1,len(timestamps)))
#			self.process_state(self.getlearner(state),t)
#
#
#
#
#
#
#
#
#class Plotter(AbstractPlotter):
#
#	def process_state(self,learner,t=None):
#
#		#cfg.dbprint('process_state called')
#
#		if t==None:t=self.timestamp()
#		X_test=self.static['X_test'][:1000]
#		Y_test=self.static['Y_test'][:1000]
#		self.static['learneractivation']=self.static['learnerinitparams'][-1]
#
#		self.remember('test loss',cfg.getlossfn()(learner.as_static()(X_test),Y_test),t)
#		self.remember('NS norm',util.norm(learning.static_NS(learner)(X_test)),t)
#		self.remember('AS norm',util.norm(learner.as_static()(X_test)),t)
#		self.remember('weight norms',[util.norm(W) for W in learner.weights[0]],t)
#		#self.remember('delta',[util.norm(W) for W in learner.weights[0]],t)
#
#	
#	def plotlosshist(self):
#		fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,7))
#		fig.suptitle(self.static['learneractivation'])
#
#		def plotax(ax):
#
#			ts=self.gethist('minibatch loss')[0]
#			ax.scatter(*cfg.times_to_ordinals(ts,*self.gethist('minibatch loss')),color='r',label='training loss',s=.3,alpha=.3)
#			ax.plot(*cfg.times_to_ordinals(ts,*self.gethist('test loss')),'bo-',label='test loss',markersize=3,lw=1)
#			ax.legend()
#			ax.set_xlabel('minibatches')
#
#			# ax.scatter(*self.gethist('minibatch loss'),color='r',label='training loss',s=.3,alpha=.3)
#			# ax.plot(*self.gethist('test loss'),'bo-',label='test loss',markersize=3,lw=1)
#			# ax.legend()
#			# ax.set_xlabel('seconds')
#
#		plotax(ax1)
#		plotax(ax2)
#		ax1.set_ylim(0,1)
#		ax2.set_yscale('log')
#		ax2.grid(True,which='major',ls='-',axis='y')
#		ax2.grid(True,which='minor',ls=':',axis='y')
#		cfg.savefig_('losses.pdf',fig=fig)
#
#	def plotweightnorms(self):
#		fig,ax=plt.subplots()
#		fig.suptitle(self.static['learneractivation'])
#
#		all_ts=self.gethist('minibatch loss')[0]
#		ts,tslices=self.gethist('weight norms')
#		wnorms=zip(*tslices)
#		
#		colors=['r','g','b','m']*5
#		for i,wnorm in enumerate(wnorms):
#			ax.plot(*cfg.times_to_ordinals(all_ts,ts,wnorm),'o-',color=colors[i],label='layer {} weights'.format(i+1),markersize=2,lw=1)
#		ax.set_xlabel('minibatches')
#		ax.legend()	
#		cfg.savefig_('weightnorms.pdf',fig=fig)
#
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
#
#	def plotfn(self,staticlearner,figname='fnplot'):
#		fig=getfnplot(self.static['sections'],staticlearner)
#		cfg.savefig_(figname+'.pdf',fig=fig)
#		plt.close('all')
#
#	def learningplots(self):
#		self.plotlosshist()
#		self.plotweightnorms()
#		#self.plot3()
#		plt.close('all')
#
#
#class DynamicPlotter(Plotter):
#
#	def __init__(self,lcls,statics,trackedvars):
#		super().__init__()
#		for name in statics:
#			#self.static[name]=cfg.getval(name)
#			self.static[name]=lcls[name]
#
#		for name,hist in trackedvars.items():
#			self.hists[name]=hist #cfg.sessionstate.linkentry(name)
#
#		#cfg.dblog('dp')
#		#self.weightshist=self.static['weightshistpointer']		
#
#"""
##class LoadedPlotter(cfg.LoadedState,Plotter):
##
##	def __init__(self,path):
##		super().__init__(path)
##		self.loadlearnerclone()
##		#self.weightshist=self.static['weightshistpointer']
#"""
#
#
#class CompPlotter():
#	def __init__(self,datapaths):
#		self.plotters={ac:LoadedPlotter(datapaths[ac]) for ac in ['ReLU','tanh']}
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
##	def plot3(self):
##		ts,tslices=self.gethist('weight norms')
##		_,fnorm=self.gethist('NS norm')
##		_,losses=self.gethist('test loss')
##		weightnorms=[max([weights for weights in lweights]) for lweights in tslices]
##
##
##		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
##		fig.suptitle(self.static['learneractivation'])
##
##		ax1.plot(weightnorms,jnp.sqrt(jnp.array(losses)),'bo-',markersize=2,lw=1)
##		ax1.set_xlabel('weights')
##		ax1.set_ylabel('l2 loss')
##		ax1.annotate('start',(weightnorms[0],jnp.sqrt(losses[0])))
##		ax1.annotate('end',(weightnorms[-1],jnp.sqrt(losses[-1])))
##
##		ax2.plot(fnorm,jnp.sqrt(jnp.array(losses)),'bo-',markersize=2,lw=1)
##		ax2.set_xlabel('||f||')
##		ax2.set_ylabel('l2 loss')
##		ax2.annotate('start',(fnorm[0],jnp.sqrt(losses[0])))
##		ax2.annotate('end',(fnorm[-1],jnp.sqrt(losses[-1])))
##
##		ax3.plot(weightnorms,fnorm,'bo-',markersize=2,lw=1)
##		ax3.set_xlabel('weights')
##		ax3.set_ylabel('||f||')
##		ax3.annotate('start',(weightnorms[0],fnorm[0]))
##		ax3.annotate('end',(weightnorms[-1],fnorm[-1]))
##
##		cfg.savefig_('plot3.pdf',fig=fig)
#
#	def plotfn(self,staticlearner,figname='fnplot'):
#		fig=getfnplot(self.static['sections'],staticlearner)
#		cfg.savefig_(figname+'.pdf',fig=fig)
#		plt.close('all')
#
#	def learningplots(self):
#		self.plotlosshist()
#		self.plotweightnorms()
#		self.plot3()
#		plt.close('all')
#
#
##class DynamicPlotter(Plotter):
##
##	def __init__(self,lcls,statics,trackedvars):
##		super().__init__()
##		for name in statics:
##			#self.static[name]=cfg.getval(name)
##			self.static[name]=lcls[name]
##		for name in trackedvars:
##			self.hists[name]=cfg.sessionstate.linkentry(name)
##
##
##class LoadedPlotter(cfg.LoadedState,Plotter):
##
##	def __init__(self,path):
##		super().__init__(path)
##		self.loadlearnerclone()
##
##
##
##
##class CompPlotter():
##	def __init__(self,datapaths):
##		self.plotters={ac:LoadedPlotter(datapaths[ac]) for ac in activations}
##
##	def prep(self,schedule):
##		for ac,plotter in self.plotters.items():
##			plotter.filtersnapshots(schedule)
##			plotter.prep()
##
##	def compareweightnorms(self):
##		fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,7))
##
##		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
##		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
##		rw1norms,rw2norms,*_=zip(*rtslices)
##		tw1norms,tw2norms,*_=zip(*ttslices)
##
##		ax1.set_title('layer 1')
##		ax1.plot(rts,rw1norms,'bo-',label='ReLU',markersize=2,lw=1)
##		ax1.plot(tts,tw1norms,'rd--',label='tanh',markersize=2,lw=1)
##
##		ax2.set_title('layer 2')
##		ax2.plot(rts,rw2norms,'bo-',label='ReLU',markersize=2,lw=1)
##		ax2.plot(tts,tw2norms,'rd--',label='tanh',markersize=2,lw=1)
##
##		ax1.legend()	
##		ax2.legend()	
##		cfg.savefig_('weightcomp.pdf',fig=fig)
##
##
##	def comp3(self):
##		rts,rtslices=self.plotters['ReLU'].gethist('weight norms')
##		tts,ttslices=self.plotters['tanh'].gethist('weight norms')
##		_,rfnorm=self.plotters['ReLU'].gethist('NS norm')
##		_,tfnorm=self.plotters['tanh'].gethist('NS norm')
##		_,rlosses=self.plotters['ReLU'].gethist('test loss')
##		_,tlosses=self.plotters['tanh'].gethist('test loss')
##		rweightnorms=[np.sqrt(x**2+y**2) for x,y,*_ in rtslices]
##		tweightnorms=[np.sqrt(x**2+y**2) for x,y,*_ in ttslices]
##
##
##		fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
##
##		ax1.plot(rweightnorms,jnp.sqrt(jnp.array(rlosses)),'bo-',markersize=2,lw=1,label='ReLU')
##		ax1.plot(tweightnorms,jnp.sqrt(jnp.array(tlosses)),'rd:',markersize=2,lw=1,label='tanh')
##		ax1.set_xlabel('weights')
##		ax1.set_ylabel('l2 loss')
##		ax1.annotate('start',(rweightnorms[0],jnp.sqrt(rlosses[0])))
##		ax1.annotate('end',(rweightnorms[-1],jnp.sqrt(rlosses[-1])))
##		ax1.annotate('start',(tweightnorms[0],jnp.sqrt(tlosses[0])))
##
#
#
