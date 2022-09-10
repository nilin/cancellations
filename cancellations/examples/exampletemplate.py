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

from ..utilities import tracking,math as mathutil,config as cfg
from ..functions import functions
import os




def train(run,learner,X_train,Y_train,**kw):

	iterations=kw['iterations']
	trainer=learning.Trainer(learner,X_train,Y_train,memory=run,**kw) 
	regsched=cfg.Scheduler(cfg.nonsparsesched(iterations,start=100))
	plotsched=cfg.Scheduler(cfg.sparsesched(iterations,start=500))
	trainer.prepnextepoch(permute=False)
	ld,_=addlearningdisplay(run,cfg.currentprocess().display)

	stopwatch1=cfg.Stopwatch()
	stopwatch2=cfg.Stopwatch()

	for i in range(iterations+1):

		loss=trainer.step()
		for mem in [run.unprocessed,run]:
			mem.addcontext('minibatchnumber',i)
			mem.remember('minibatch loss',loss)

		if regsched.activate(i):
			run.unprocessed.remember('weights',learner.weights)
			cfg.save(run.unprocessed,run.outpath+'data/unprocessed',echo=False)
			cfg.write('loss {:.3f}, iterations: {}'.format(loss,i),run.outpath+'metadata.txt',mode='w')	

		if plotsched.activate(i):
			fplot()
			lplot()

		if stopwatch1.tick_after(.05):
			ld.draw()

		if stopwatch2.tick_after(.5):
			if cfg.act_on_input(cfg.checkforinput())=='b': break



# backend
####################################################################################################

#def inspect():
#	testantisymmetry(cprof.target,cprof.learner,cprof.genX(100))
#	functions.inspect(cprof.target,cprof.genX(55),msg='target')
#	functions.inspect(cprof.learner,cprof.genX(55),msg='learner')


def testantisymmetry(target,learner,X):
	cfg.logcurrenttask('verifying antisymmetry of target')
	testing.verify_antisymmetric(target.eval,X[:100])
	cfg.logcurrenttask('verifying antisymmetry of learner')
	testing.verify_antisymmetric(learner.eval,X[:100])
	cfg.clearcurrenttask()
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
	run=cfg.currentprocess()
	return 'n={}, target: {}{}learner: {}'.format(run['n'],\
		run.target.richtypename(),separator,run.learner.richtypename())

def INFO(separator='\n\n',width=100):
	run=cfg.currentprocess()
	globals().update(cfg.params)
	targetinfo='target\n\n{}'.format(cfg.indent(run.target.getinfo()))
	learnerinfo='learner\n\n{}'.format(cfg.indent(run.learner.getinfo()))
	return cfg.wraptext(targetinfo+'\n'*4+learnerinfo)








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
	cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)



	fig,ax=plt.subplots()
	ax.set_title('performance '+info())
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	cfg.savefig('{}{}'.format(cfg.outpath,'performance.pdf'),fig=fig)


process_snapshot=process_snapshot_0
plotexample=plotexample_0

def processandplot(unprocessed,pfunc,X,Y,process_snapshot_fn=None,plotexample_fn=None):

	pfunc=pfunc.getemptyclone()
	if process_snapshot_fn==None: process_snapshot_fn=process_snapshot
	if plotexample_fn==None: plotexample_fn=plotexample
	processed=cfg.Memory()

	weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')
	for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):

		if cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))=='b': break
		process_snapshot(processed,mathutil.fixparams(pfunc.f,weights),X,Y,i)		

	plotexample(unprocessed,processed)
	cfg.clearcurrenttask()
	return processed

def lplot():
	run=cfg.currentprocess()
	processandplot(run.unprocessed,run.learner,run.X_test,run.Y_test)


# function plots

def plotfunctions(sections,f,figtitle,path):
	plt.close('all')
	for fignum,section in enumerate(sections):
		fig=section.plot_y_vs_f_SI(f)
		if cfg.trackcurrenttask('generating function plots',(fignum+1)/len(sections))=='b': break
		fig.suptitle(figtitle)
		cfg.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
	cfg.clearcurrenttask()

def fplot():
	run=cfg.currentprocess()
	figtitle=info(separator='\n')
	figpath='{}{} minibatches'.format(run.outpath,int(run.unprocessed.getval('minibatchnumber')))
	plotfunctions(run.sections,run.learner.eval,figtitle,figpath)


# dashboard
####################################################################################################

def act_on_input(key):
	if key=='q': quit()
	if key=='l': lplot()
	if key=='f': fplot()
	if key=='o': cfg.showfile(cfg.getoutpath())
	return key






def prepdisplay(run):

	display=run.display

	# columndisplay

	instructions=run.instructions+'\n\n\nPress [l] (lowercase L) to generate learning plots.\n'+\
		'Press [f] to generate functions plot.\nPress [o] to open output folder.\
		\n\nPress [b] to break from current task.\nPress [q] to quit. '

	a,b,c,d=5,display.width//2-5,display.width//2+5,display.width-5
	y0=3


	column1=cdisplay.ConcreteDisplay(xlim=(a,b),ylim=(y0,3*display.height//4),memory=run)
	column1.add(disp.StaticText(msg=instructions))
	column1.add(disp.VSpace(2))
	column1.add(disp.Hline())
	column1.add(disp.LogDisplay(height=10))
	column1.add(disp.Hline())
	column1.add(disp.RunText(query='currenttask',msgtransform=lambda msg:msg if run.getcurrenttask()!=None else ''))
	column1.add(disp.Bar('currenttaskcompleteness',msgtransform=lambda msg:msg if run.getcurrenttask()!=None else ''))
	column1.draw()

	run.addlistener(column1,'recentlog')
	run.addlistener(column1,'currenttaskcompleteness')
	run.addlistener(column1,'target |Af|')


	column2=cdisplay.ConcreteDisplay(xlim=(c,d),ylim=(y0,3*display.height//4))
	column2.add(disp.RunText(query='runinfo',wrap=True))

	run.addlistener(column2,'runinfo')


	display.add(column1,'column1')
	display.add(column2,'column2')






def addlearningdisplay(run,display):
	import cdisplay

	a,b=5,display.width-5

	ld=cdisplay.ConcreteDisplay((a,b),(display.height-10,display.height-1))
	ld.add(disp.NumberPrint('minibatch loss',msg='training loss {:.3}'))
	ld.add(disp.Bar('minibatch loss',style='.',emptystyle=' '))
	ld.add(disp.Bar('minibatch loss',style=disp.BOX,avg_of=25))
	ld.add(disp.VSpace(1))
	ld.add(disp.NumberPrint('minibatchnumber',msg='minibatch number {:.0f}'))

	return run.display.add(ld,'learningdisplay')
	
	



