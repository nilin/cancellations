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
import learning
import plottools as pt
import matplotlib.pyplot as plt
import util
import display as disp
import config as cfg
from config import session
import testing
import functions






def runexample(prep_and_run):
	cfg.trackduration=True

	if 'debug' in cfg.cmdparams:
		import debug
		disp.clear()
		get_pynputkeyboard()
		prep_and_run()

	else:
		cfg.prepdashboard=prepdashboard
		import cdisplay
		cdisplay.RID(prep_and_run)



def register(*names,sourcedict):
	globals().update({k:sourcedict[k] for k in names})



def train(learner,X_train,Y_train,**kw):

	iterations=kw['iterations']
	trainer=learning.Trainer(learner,X_train,Y_train,memory=session,**kw) 
	cfg.regsched=cfg.Scheduler(cfg.nonsparsesched(iterations,start=100))
	cfg.provide(plotsched=cfg.Scheduler(cfg.sparsesched(iterations,start=500)))
	trainer.prepnextepoch(permute=False)
	addlearningdisplay()


	for i in range(iterations+1):
		try: cfg.learningdisplay.draw()
		except: pass

		loss=trainer.step()
		for mem in [cfg.unprocessed,session]:
			mem.addcontext('minibatchnumber',i)
			mem.remember('minibatch loss',loss)

		if cfg.regsched.activate(i):
			cfg.unprocessed.remember('weights',learner.weights)
			cfg.save(cfg.unprocessed,cfg.outpath+'data/unprocessed',echo=False)

		if cfg.plotsched.activate(i):
			fplot()
			lplot()

		if i%50==0:
			if cfg.act_on_input(cfg.getinput())=='b': break



# backend
####################################################################################################

def inspect():
	testantisymmetry(cfg.target,cfg.learner,cfg.genX(100))
	functions.inspect(cfg.target,cfg.genX(55),msg='target')
	functions.inspect(cfg.learner,cfg.genX(55),msg='learner')


def testantisymmetry(target,learner,X):
	cfg.logcurrenttask('verifying antisymmetry of target')
	testing.verify_antisymmetric(target.eval,X[:100])
	cfg.logcurrenttask('verifying antisymmetry of learner')
	testing.verify_antisymmetric(learner.eval,X[:100])
	cfg.clearcurrenttask()
	return True


def adjustnorms(Afdescr,X,iterations=500,**learningparams):
	Af=Afdescr.f
	f=functions.switchtype(Afdescr).f
	normratio=jax.jit(lambda weights,X:util.norm(f(weights,X))/util.norm(Af(weights,X)))
	weights=Afdescr.weights

	cfg.log('|f|/|Af|={:.3f}, |Af|={:.3f} before adjustment'.format(\
		normratio(weights,X[:1000]),util.norm(Af(weights,X[:1000]))))

	@jax.jit
	def directloss(params,Y):
		Af_norm=util.norm(Af(params,Y))
		f_norm=util.norm(f(params,Y))
		normloss=jnp.abs(jnp.log(Af_norm))
		ratioloss=jnp.log(f_norm/Af_norm)
		return normloss+ratioloss

	trainer=learning.DirectlossTrainer(directloss,weights,X,**learningparams)

	try:
		temp1=cfg.statusdisplay.add(disp.NumberPrint('target |f|/|Af|',msg='\n\n|f|/|Af|={:.3f} (objective: decrease)'))
		temp2=cfg.statusdisplay.add(disp.RplusBar('target |f|/|Af|'))
		temp3=cfg.statusdisplay.add(disp.NumberPrint('target |Af|',msg='\n|Af|={:.3f} (objective: approach 1)'))
		temp4=cfg.statusdisplay.add(disp.RplusBar('target |Af|'))
	except: pass
	
	for i in range(iterations):
		trainer.step()
		cfg.session.trackcurrent('target |Af|',util.norm(Af(trainer.learner.weights,X[:100])))
		cfg.session.trackcurrent('target |f|/|Af|',normratio(trainer.learner.weights,X[:100]))
		temp1.draw()
		if cfg.getinput()=='b':break

	try: cfg.statusdisplay.delete(temp1,temp2,temp3,temp4)
	except: pass

	weights=trainer.learner.weights
	cfg.log('|f|/|Af|={:.3f}, |Af|={:.3f} after adjustment'.format(\
		normratio(weights,X[:1000]),util.norm(Af(weights,X[:1000]))))
	return weights


# info
####################################################################################################


def info(separator=' | '):
	globals().update(cfg.params)
	return 'n={}, target: {}{}learner: {}'.format(n,cfg.target.richtypename(),separator,cfg.learner.richtypename())

def INFO(separator='\n\n',width=100):
	globals().update(cfg.params)
	targetinfo='target\n\n{}'.format(cfg.indent(cfg.target.getinfo()))
	learnerinfo='learner\n\n{}'.format(cfg.indent(cfg.learner.getinfo()))
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
	cfg.clearcurrenttask()
	return processed

def lplot():
	processandplot(cfg.unprocessed,cfg.learner,cfg.X_test,cfg.Y_test)


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
	figtitle=info()
	figpath='{}{} minibatches'.format(cfg.outpath,int(cfg.unprocessed.getval('minibatchnumber')))
	plotfunctions(cfg.sections,cfg.learner.eval,figtitle,figpath)


# dashboard
####################################################################################################

def act_on_input(key):
	if key=='q': quit()
	if key=='l': lplot()
	if key=='f': fplot()
	if key=='o':
		cfg.showfile(cfg.getoutpath())
	return key

cfg.act_on_input=act_on_input




def prepdashboard():

	import cdisplay

	# columndisplay

	instructions=cfg.instructions+'\n\n\nPress [l] (lowercase L) to generate learning plots.\n'+\
		'Press [f] to generate functions plot.\nPress [o] to open output folder.\
		\n\nPress [b] to break from current task.\nPress [q] to quit. '

	DB=cfg.dashboard
	w=DB.width; h=DB.height
	a,b,c,d=5,w//2-5,w//2+5,w-5
	y0,y1=3,h-15


	# cdisplay1

	CD1=cdisplay.ConcreteDisplay(xlim=(a,b),ylim=(y0,y1))

	instructiondisplay=disp.StaticText(msg=instructions)
	logdisplay=disp.LogDisplay(height=10)
	statusdisplay=disp.StackedDisplay(memory=session)
	statusdisplay.add(disp.SessionText(query='currenttask',msgtransform=lambda msg:msg if cfg.getcurrenttask()!=None else ''))
	statusdisplay.add(disp.Bar('currenttaskcompleteness',msgtransform=lambda msg:msg if cfg.getcurrenttask()!=None else ''))
	cfg.statusdisplay=statusdisplay

	CD1.add(instructiondisplay)
	CD1.add(disp.VSpace(2))
	CD1.add(disp.Hline())
	CD1.add(logdisplay)
	CD1.add(disp.Hline())
	CD1.add(disp.VSpace(2))
	CD1.add(statusdisplay)


	# infodisplay

	CD2=cdisplay.ConcreteDisplay(xlim=(c,d),ylim=(y0,y1))
	CD2.add(disp.SessionText(query='sessioninfo',wrap=True))


	session.addlistener(CD1,'recentlog')
	session.addlistener(CD1,'currenttaskcompleteness')
	session.addlistener(CD1,'target |Af|')
	session.addlistener(CD2,'sessioninfo')


	cfg.dashboard.add(CD1,'column')
	cfg.dashboard.add(CD2,'info')






def addlearningdisplay():
	import cdisplay
	DB=cfg.dashboard

	w=DB.width; h=DB.height
	a,b=5,w-5

	CD3=cdisplay.ConcreteDisplay((a,b),(h-10,h-1))
	CD3.add(disp.NumberPrint('minibatch loss',msg='training loss {:.3}'))
	CD3.add(disp.Bar('minibatch loss',style='.',emptystyle=' '))
	CD3.add(disp.Bar('minibatch loss',style=disp.BOX,avg_of=25))
	CD3.add(disp.VSpace(1))
	CD3.add(disp.NumberPrint('minibatchnumber',msg='minibatch number {:.0f}'))

	cfg.learningdisplay=CD3
	return cfg.dashboard.add(CD3)




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