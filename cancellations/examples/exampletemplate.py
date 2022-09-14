#
# nilin
# 
# 2022/7
#


from cgi import test
from re import I
import jax.numpy as jnp
import jax.random as rnd
import jax
from ..learning import learning
from . import plottools as pt
import matplotlib.pyplot as plt
from ..display import cdisplay,display as disp

from ..utilities import arrayutil as mathutil, tracking,config as cfg,sysutil,textutil
from ..functions import functions
from ..learning import testing
import os




def train(run,learner,X_train,Y_train,**kw):

	iterations=kw['iterations']
	trainer=learning.Trainer(learner,X_train,Y_train,memory=run,**kw) 
	regsched=tracking.Scheduler(tracking.nonsparsesched(iterations,start=100))
	plotsched=tracking.Scheduler(tracking.sparsesched(iterations,start=1000))
	trainer.prepnextepoch(permute=False)
	ld,_=addlearningdisplay(run,tracking.currentprocess().display)

	stopwatch1=tracking.Stopwatch()
	stopwatch2=tracking.Stopwatch()

	for i in range(iterations+1):

		loss=trainer.step()
		for mem in [run.unprocessed,run]:
			mem.addcontext('minibatchnumber',i)
			mem.remember('minibatch loss',loss)

		if regsched.activate(i):
			run.unprocessed.remember('weights',learner.weights)
			sysutil.save(run.unprocessed,run.outpath+'data/unprocessed',echo=False)
			sysutil.write('loss={:.3f} iterations={} n={} d={}'.format(loss,i,run.n,run.d),run.outpath+'metadata.txt',mode='w')	

		if plotsched.activate(i):
			fplot()
			lplot()

		if stopwatch1.tick_after(.05):
			ld.draw()

		if stopwatch2.tick_after(.5):
			if tracking.act_on_input(tracking.checkforinput())=='b': break

	return trainer



# backend
####################################################################################################

#def inspect():
#	testantisymmetry(cprof.target,cprof.learner,cprof.genX(100))
#	functions.inspect(cprof.target,cprof.genX(55),msg='target')
#	functions.inspect(cprof.learner,cprof.genX(55),msg='learner')


def testantisymmetry(target,learner,X):
	tracking.logcurrenttask('verifying antisymmetry of target')
	testing.verify_antisymmetric(target.eval,X[:100])
	tracking.logcurrenttask('verifying antisymmetry of learner')
	testing.verify_antisymmetric(learner.eval,X[:100])
	tracking.clearcurrenttask()
	return True


def adjustnorms(Afdescr,X,iterations=500,**learningparams):
	run=tracking.currentprocess()
	Af=Afdescr.f
	f=functions.switchtype(Afdescr).f
	normratio=jax.jit(lambda weights,X:mathutil.norm(f(weights,X))/mathutil.norm(Af(weights,X)))
	weights=Afdescr.weights

	tracking.log('|f|/|Af|={:.3f}, |Af|={:.3f} before adjustment'.format(\
		normratio(weights,X[:1000]),mathutil.norm(Af(weights,X[:1000]))))

	@jax.jit
	def directloss(params,Y):
		Af_norm=mathutil.norm(Af(params,Y))
		f_norm=mathutil.norm(f(params,Y))
		normloss=jnp.abs(jnp.log(Af_norm))
		ratioloss=jnp.log(f_norm/Af_norm)
		return normloss+ratioloss

	trainer=learning.DirectlossTrainer(directloss,weights,X,**learningparams)

	_,key1=run.display.column1.add(disp.NumberPrint('target |f|/|Af|',msg='\n\n|f|/|Af|={:.3f} (objective: decrease)'))
	_,key2=run.display.column1.add(disp.RplusBar('target |f|/|Af|'))
	_,key3=run.display.column1.add(disp.NumberPrint('target |Af|',msg='\n|Af|={:.3f} (objective: approach 1)'))
	_,key4=run.display.column1.add(disp.RplusBar('target |Af|'))
	
	for i in range(iterations):
		trainer.step()
		run.trackcurrent('target |Af|',mathutil.norm(Af(trainer.learner.weights,X[:100])))
		run.trackcurrent('target |f|/|Af|',normratio(trainer.learner.weights,X[:100]))
		if tracking.stopwatch.tick_after(.05) and tracking.act_on_input(tracking.checkforinput())=='b':break

	run.display.column1.delkeys(key1,key2,key3,key4)

	weights=trainer.learner.weights
	tracking.log('|f|/|Af|={:.3f}, |Af|={:.3f} after adjustment'.format(\
		normratio(weights,X[:1000]),mathutil.norm(Af(weights,X[:1000]))))
	return weights


# info
####################################################################################################


def info(separator=' | '):
	run=tracking.currentprocess()
	return 'n={}, target: {}{}learner: {}'.format(run['n'],\
		run.target.richtypename(),separator,run.learner.richtypename())

def INFO(separator='\n\n',width=100):
	run=tracking.currentprocess()
	targetinfo='target\n\n{}'.format(textutil.indent(run.target.getinfo()))
	learnerinfo='learner\n\n{}'.format(textutil.indent(run.learner.getinfo()))
	return disp.wraptext(targetinfo+'\n'*4+learnerinfo)








# plots
####################################################################################################

# learning plots

def process_snapshot_0(processed,f,X,Y,i):
	processed.addcontext('minibatchnumber',i)
	processed.remember('Af norm',jnp.average(f(X[:100])**2))
	processed.remember('test loss',mathutil.SI_loss(f(X),Y))

def plotexample_0(unprocessed,processed):
	plt.close('all')


	fig,(ax0,ax1)=plt.subplots(2)
	fig.suptitle('test loss '+info())

	ax0.plot(*mathutil.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
	ax0.legend()
	ax0.set_ylim(bottom=0,top=1)
	ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	ax1.plot(*mathutil.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
	ax1.legend()
	ax1.set_yscale('log')
	ax1.grid(True,which='major',ls='-',axis='y')
	ax1.grid(True,which='minor',ls=':',axis='y')
	sysutil.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)



	fig,ax=plt.subplots()
	ax.set_title('performance '+info())
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	sysutil.savefig('{}{}'.format(cfg.outpath,'performance.pdf'),fig=fig)


process_snapshot=process_snapshot_0
plotexample=plotexample_0

def processandplot(unprocessed,pfunc,X,Y,process_snapshot_fn=None,plotexample_fn=None):

	pfunc=pfunc.getemptyclone()
	if process_snapshot_fn==None: process_snapshot_fn=process_snapshot
	if plotexample_fn==None: plotexample_fn=plotexample
	processed=tracking.Memory()

	weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')
	for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):

		if tracking.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))=='b': break
		process_snapshot(processed,mathutil.fixparams(pfunc.f,weights),X,Y,i)		

	plotexample(unprocessed,processed)
	tracking.clearcurrenttask()
	return processed

def lplot():
	run=tracking.currentprocess()
	processandplot(run.unprocessed,run.learner,run.X_test,run.Y_test)


# function plots

def plotfunctions(sections,f,figtitle,path):
	plt.close('all')
	for fignum,section in enumerate(sections):
		fig=section.plot(f)
		if tracking.trackcurrenttask('generating function plots',(fignum+1)/len(sections))=='b': break
		fig.suptitle(figtitle+'\n\n'+section.info)
		sysutil.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
	tracking.clearcurrenttask()

def fplot():
	run=tracking.currentprocess()
	figtitle=info(separator='\n')
	figpath='{}{} minibatches'.format(run.outpath,int(run.unprocessed.getval('minibatchnumber')))
	plotfunctions(run.sections,run.learner.eval,figtitle,figpath)


# dashboard
####################################################################################################

def act_on_input(key):
	if key=='q': quit()
	if key=='l': lplot()
	if key=='f': fplot()
	if key=='o': sysutil.showfile(tracking.getoutpath())
	return key






def prepdisplay(run):

	display=run.display

	# columndisplay

	instructions=run.name+'\n\n\nPress [l] (lowercase L) to generate learning plots.\n'+\
		'Press [f] to generate functions plot.\nPress [o] to open output folder.\
		\n\nPress [b] to break from current task.\nPress [q] to quit. '
	
	x0,x2=display.xlim
	x1=(x0+x2)//2
	y0,y2=display.ylim
	y1=(y0+3*y2)//4


	column1=cdisplay.ConcreteDisplay(xlim=(x0,x1-5),ylim=(y0,y1),memory=run)
	column1.add(disp.StaticText(msg=instructions))
	column1.add(disp.VSpace(2))
	column1.add(disp.Hline())
	column1.add(disp.LogDisplay(height=10))
	column1.add(disp.Hline())
#	column1.add(disp.RunText(query='currenttask',msgtransform=lambda msg:msg if run.getcurrenttask()!=None else ''))
#	column1.add(disp.Bar('currenttaskcompleteness',msgtransform=lambda msg:msg if run.getcurrenttask()!=None else ''))
	column1.draw()

	run.addlistener(column1,'recentlog')
#	run.addlistener(column1,'currenttaskcompleteness')
#	run.addlistener(column1,'target |Af|')


	column2=cdisplay.ConcreteDisplay(xlim=(x1+5,x2),ylim=(y0,y1))
	run.infodisplay,_=column2.add(disp.StaticText(msg='',wrap=True))

#	run.addlistener(column2,'runinfo')


	display.add(column1,'column1')
	display.add(column2,'column2')






def addlearningdisplay(run,display):
	from ..display import cdisplay

	a,b=display.xlim[0]+2,display.xlim[1]-2

	ld=cdisplay.ConcreteDisplay((a,b),(display.height-6,display.height-1))
	ld.add(disp.NumberPrint('minibatch loss',msg='training loss {:.2E}',avg_of=100))
	ld.add(disp.Bar('minibatch loss',style=textutil.dash,emptystyle=' ',avg_of=1))
	ld.add(disp.Bar('minibatch loss',style=disp.BOX,emptystyle=' ',avg_of=10))
	ld.add(disp.Bar('minibatch loss',style=disp.BOX,emptystyle='_',avg_of=100))
	ld.add(disp.VSpace(1))
	ld.add(disp.NumberPrint('minibatchnumber',msg='minibatch number {:.0f}'))

	return run.display.add(ld,'learningdisplay')
	
	



