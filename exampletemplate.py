#
# nilin
# 
# 2022/7
#


import jax.numpy as jnp
import jax.random as rnd
import jax
import learning
import plottools as pt
import matplotlib.pyplot as plt
import util
import dashboard as db
import config as cfg
from config import act_on_input, getinput, session
import testing
import functions






def runexample(prep):
	cfg.trackduration=True

	def prep_and_run():
		target,learner=prep()
		runfn(target,learner)

	if 'debug' in cfg.cmdparams:
		import debug
		db.clear()
		get_pynputkeyboard()
		prep_and_run()

	else:
		import run_in_display
		run_in_display.RID(prep_and_run)


def runfn(target,learner):
	globals().update(cfg.params)

	global unprocessed,X,X_test,Y,Y_test,sections,_learner_,_target_
	_target_,_learner_=target,learner

	cfg.sessioninfo='sessionID: {}\n{}'.format(cfg.sessionID,INFO())
	cfg.write(cfg.sessioninfo,cfg.outpath+'info.txt',mode='w')
	addinfodisplay(cfg.sessioninfo)

	
	cfg.currentkeychain=2
	X=cfg.genX(samples_train)
	X_test=cfg.genX(samples_test)

	cfg.logcurrenttask('preparing training data')
	Y=target.eval(X)
	cfg.logcurrenttask('preparing test data')
	Y_test=target.eval(X_test)

	setupdata={k:globals()[k] for k in ['X_test','Y_test']}|{'target':target.compress(),'learner':learner.compress()}
	cfg.save(setupdata,cfg.outpath+'data/setup')

	trainer=learning.Trainer(learner,X,Y,\
		weight_decay=weight_decay,minibatchsize=minibatchsize,lossfn=util.SI_loss) #,lossgrad=mv.gen_lossgrad(AS,lossfn=util.SI_loss))
	cfg.logcurrenttask('preparing slices for plotting')
	cfg.currentkeychain=4
	sections=pt.genCrossSections(X,Y,target.eval)

	cfg.regsched=cfg.Scheduler(cfg.nonsparsesched(iterations,start=100))
	cfg.provide(plotsched=cfg.Scheduler(cfg.sparsesched(iterations,start=1000)))

	cfg.logcurrenttask('begin training')


	unprocessed=cfg.ActiveMemory()
	addlearningdisplay(unprocessed)
	#prepdashboard(sessioninfo,unprocessed,instructions=cfg.explanation)
	for i in range(iterations+1):
		try: cfg.dashboard.draw('learning')
		except: pass


		loss=trainer.step()
		unprocessed.addcontext('minibatchnumber',i)
		unprocessed.remember('minibatch loss',loss)

		if cfg.regsched.activate(i):
			unprocessed.remember('weights',learner.weights)
			cfg.save(unprocessed,cfg.outpath+'data/unprocessed',echo=False)

		if cfg.plotsched.activate(i):
			fplot()
			lplot()

		if i%50==0:
			if cfg.act_on_input(cfg.getinput())=='b': break



# backend
####################################################################################################

def testantisymmetry(target,learner,X):
	cfg.logcurrenttask('verifying antisymmetry of target')
	testing.verify_antisymmetric(target.eval,X[:100])
	cfg.logcurrenttask('verifying antisymmetry of learner')
	testing.verify_antisymmetric(learner.eval,X[:100])
	return True


def adjustnorms(Afdescr,X,iterations=500,**learningparams):
	Af=Afdescr.f
	f=functions.switchtype(Afdescr).f
	weights=Afdescr.weights

	normratio=jax.jit(lambda weights,X:util.norm(f(weights,X))/util.norm(Af(weights,X)))
	cfg.log('before adjustment:')
	cfg.log('|f|/|Af|={:.3}'.format(normratio(weights,X)))
	cfg.log('|Af|={:.3}'.format(util.norm(Af(weights,X))))

	@jax.jit
	def directloss(params,Y):
		Af_norm=util.norm(Af(params,Y))
		f_norm=util.norm(f(params,Y))
		normloss=jnp.abs(jnp.log(Af_norm))*1.1
		ratioloss=jnp.log(f_norm/Af_norm)
		return normloss+ratioloss

	trainer=learning.DirectlossTrainer(directloss,weights,X,**learningparams)

	try:
		temp3=cfg.statusdisplay.add(db.NumberPrint('target |f|/|Af|',msg='\n\n|f|/|Af|={:.3f} objective: decrease (may be fixed by function class)'))
		temp4=cfg.statusdisplay.add(db.RplusBar('target |f|/|Af|'))
		temp1=cfg.statusdisplay.add(db.NumberPrint('target |Af|',msg='\n|Af|={:.3f} objective: ~1'))
		temp2=cfg.statusdisplay.add(db.RplusBar('target |Af|'))
	except: pass
	
	for i in range(iterations):
		if cfg.trackcurrenttask('adjusting target norm',i/iterations)=='b': break
		trainer.step()

		cfg.session.trackcurrent('target |Af|',util.norm(Af(trainer.learner.weights,X)))
		cfg.session.trackcurrent('target |f|/|Af|',normratio(trainer.learner.weights,X))

	try: cfg.statusdisplay.delete(temp1,temp2,temp3,temp4)
	except: pass

	weights=trainer.learner.weights
	cfg.log('after adjustment:')
	cfg.log('|f|/|Af|={:.3}'.format(normratio(weights,X)))
	cfg.log('|Af|={:.3}'.format(util.norm(Af(weights,X))))
	return weights


# info
####################################################################################################


def info(separator=' | '):
	globals().update(cfg.params)
	return 'n={}, target: {}{}learner: {}'.format(n,_target_.richtypename(),separator,_learner_.richtypename())

def INFO(separator='\n\n',width=100):
	globals().update(cfg.params)
	targetinfo='target\n\n{}'.format(cfg.indent(_target_.getinfo()))
	learnerinfo='learner\n\n{}'.format(cfg.indent(_learner_.getinfo()))
	#return lb+'n={}'.format(n)+lb+targetinfo+'\n'*4+learnerinfo+lb
	return cfg.wraptext(targetinfo+'\n'*4+learnerinfo)








# plots
####################################################################################################

# learning plots

def process_snapshot_0(processed,f,X,Y,i):
	processed.addcontext('minibatchnumber',i)
	processed.remember('Af norm',jnp.average(f(X[:100])**2))
	processed.remember('test loss',util.SI_loss(f(X),Y))

def plotexample_0(unprocessed,processed):
	plt.close('all')


	fig,(ax0,ax1)=plt.subplots(2)
	fig.suptitle('test loss '+info())

	ax0.plot(*util.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
	ax0.legend()
	ax0.set_ylim(bottom=0,top=1)
	ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	ax1.plot(*util.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
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

	processed=cfg.ActiveMemory()

	weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')
	for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):

		if cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))=='b': break
		process_snapshot(processed,pfunc.fwithparams(weights),X,Y,i)		


	plotexample(unprocessed,processed)
	#cfg.save(processed,cfg.outpath+'data/processed')

	cfg.clearcurrenttask()
	return processed



# function plots

def plotfunctions(sections,f,figtitle,path):
	plt.close('all')
	for fignum,section in enumerate(sections):
		fig=section.plot_y_vs_f_SI(f)
		if cfg.trackcurrenttask('generating function plots',(fignum+1)/len(sections))=='b': break
		fig.suptitle(figtitle)
		cfg.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
	cfg.clearcurrenttask()




# dashboard
####################################################################################################

def act_on_input(key):
	if key=='q': quit()
	if key=='l': lplot()
	if key=='f': fplot()
	if key=='o': cfg.showfile(cfg.getoutpath())
	return key

cfg.act_on_input=act_on_input

def lplot():
	processandplot(unprocessed,_learner_,X_test,Y_test)

def fplot():
	figtitle=info()
	figpath='{}{} minibatches'.format(cfg.outpath,int(unprocessed.getval('minibatchnumber')))
	plotfunctions(sections,_learner_.eval,figtitle,figpath)


def prepdashboard(instructions):

	try:	
		instructions='\n\nPress [l] (lowercase L) to generate learning plots.\n'+\
			'Press [f] to generate functions plot.\nPress [o] to open output folder.\
			\n\nPress [b] to break from current task.\nPress [q] to quit. '

		DB=cfg.dashboard
		w=DB.width; h=DB.height
		x1=w//2-10; x2=w//2+10
		halfw=x1

		# column
		instructiondisplay=db.StaticText(instructions)
		statusdisplay=db.StackedDisplay(memory=session,width=halfw)
		logdisplay=db.LogDisplay(height=20)
		column=db.StackedDisplay()
		column.stack(instructiondisplay,statusdisplay,logdisplay)

		statusdisplay.add(db.ValDisplay('currenttask',msg='{}'))
		statusdisplay.add(db.Bar('currenttaskcompleteness'))
		cfg.statusdisplay=statusdisplay

		col=DB.add(column,(0,x1),(0,h-11))

		session.addlistener(col,'recentlog')
		session.addlistener(col,'currenttaskcompleteness')
	
	except: cfg.log('dashboard skipped')


def addlearningdisplay(learningmemory):
	try:
		DB=cfg.dashboard
		w=DB.width; h=DB.height
		x1=w//2-10; x2=w//2+10
		halfw=x1

		learningdisplay=db.StackedDisplay(memory=learningmemory,width=w)
		learningdisplay.add(db.NumberPrint('minibatch loss',msg='training loss {:.3}'))
		learningdisplay.add(db.Bar('minibatch loss',style='.',emptystyle=' '))
		learningdisplay.add(db.Bar('minibatch loss',style=db.BOX,avg_of=25))
		learningdisplay.addspace()
		learningdisplay.add(db.NumberPrint('minibatchnumber',msg='minibatch number {:.0f}'))
		cfg.dashboard.add(learningdisplay,(0,w-1),(h-10,h-1),name='learning',draw=False)
	except: pass


def addinfodisplay(sessioninfo):
	try:
		DB=cfg.dashboard
		w=DB.width; h=DB.height
		x1=w//2-10; x2=w//2+10
		halfw=x1

		infodisplay=db.StaticText(sessioninfo)
		cfg.dashboard.add(infodisplay,(x2,w-1),(0,h-1))
	except: pass


# pynputkeyboard for debug
####################################################################################################

def get_pynputkeyboard():
	from pynput import keyboard

	cfg.inputbuffer=None
	def getinput():
		out=cfg.inputbuffer
		cfg.inputbuffer=None
		return out
	cfg.getinput=getinput

	def on_press(key):
		cfg.inputbuffer=key
	listener=keyboard.Listener(on_press=on_press)
	listener.start()