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





def train(learner,X_train,Y_train,**kw):
	cprof=cfg.currentprofile()

	iterations=kw['iterations']
	trainer=learning.Trainer(learner,X_train,Y_train,memory=cprof.run,**kw) 
	cprof.regsched=cfg.Scheduler(cfg.nonsparsesched(iterations,start=100))
	cprof.plotsched=cfg.Scheduler(cfg.sparsesched(iterations,start=500))
	trainer.prepnextepoch(permute=False)
	addlearningdisplay()


	for i in range(iterations+1):
		try: cprof.learningdisplay.draw()
		except: pass

		loss=trainer.step()
		for mem in [cprof.unprocessed,cprof.run]:
			mem.addcontext('minibatchnumber',i)
			mem.remember('minibatch loss',loss)

		if cprof.regsched.activate(i):
			cprof.unprocessed.remember('weights',learner.weights)
			cfg.save(cprof.unprocessed,cprof.outpath+'data/unprocessed',echo=False)

		if cprof.plotsched.activate(i):
			fplot()
			lplot()

		if cfg.stopwatch.tick_after(.2) and cfg.currentprofile().act_on_input(cfg.getinput())=='b':
			break



# backend
####################################################################################################

def inspect():
	cprof=cfg.currentprofile()
	testantisymmetry(cprof.target,cprof.learner,cprof.genX(100))
	functions.inspect(cprof.target,cprof.genX(55),msg='target')
	functions.inspect(cprof.learner,cprof.genX(55),msg='learner')


def testantisymmetry(target,learner,X):
	cfg.logcurrenttask('verifying antisymmetry of target')
	testing.verify_antisymmetric(target.eval,X[:100])
	cfg.logcurrenttask('verifying antisymmetry of learner')
	testing.verify_antisymmetric(learner.eval,X[:100])
	cfg.clearcurrenttask()
	return True


def adjustnorms(Afdescr,X,iterations=500,**learningparams):
	cprof=cfg.currentprofile()
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

	temp1=cprof.statusdisplay.add(disp.NumberPrint('target |f|/|Af|',msg='\n\n|f|/|Af|={:.3f} (objective: decrease)'))
	temp2=cprof.statusdisplay.add(disp.RplusBar('target |f|/|Af|'))
	temp3=cprof.statusdisplay.add(disp.NumberPrint('target |Af|',msg='\n|Af|={:.3f} (objective: approach 1)'))
	temp4=cprof.statusdisplay.add(disp.RplusBar('target |Af|'))
	
	for i in range(iterations):
		trainer.step()
		cprof.run.trackcurrent('target |Af|',util.norm(Af(trainer.learner.weights,X[:100])))
		cprof.run.trackcurrent('target |f|/|Af|',normratio(trainer.learner.weights,X[:100]))
		if cfg.stopwatch.tick_after(.05) and cfg.getinput()=='b':break

	cprof.statusdisplay.delete(temp1,temp2,temp3,temp4)

	weights=trainer.learner.weights
	cfg.log('|f|/|Af|={:.3f}, |Af|={:.3f} after adjustment'.format(\
		normratio(weights,X[:1000]),util.norm(Af(weights,X[:1000]))))
	return weights


# info
####################################################################################################


def info(separator=' | '):
	cprof=cfg.currentprofile()
	return 'n={}, target: {}{}learner: {}'.format(cprof['n'],\
		cprof['target'].richtypename(),separator,cprof['learner'].richtypename())

def INFO(separator='\n\n',width=100):
	cprof=cfg.currentprofile()
	globals().update(cfg.params)
	targetinfo='target\n\n{}'.format(cfg.indent(cprof['target'].getinfo()))
	learnerinfo='learner\n\n{}'.format(cfg.indent(cprof['learner'].getinfo()))
	return cfg.wraptext(targetinfo+'\n'*4+learnerinfo)








# plots
####################################################################################################

# learning plots

def process_snapshot_0(processed,f,X,Y,i):
	processed.addcontext('minibatchnumber',i)
	processed.remember('Af norm',jnp.average(f(X[:100])**2))
	processed.remember('test loss',util.SI_loss(f(X),Y))

def plotexample_0(unprocessed,processed):
	cprof=cfg.currentprofile()
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
	cfg.savefig('{}{}'.format(cprof.outpath,'losses.pdf'),fig=fig)



	fig,ax=plt.subplots()
	ax.set_title('performance '+info())
	I,t=unprocessed.gethistbytime('minibatchnumber')
	ax.plot(t,I)
	ax.set_xlabel('time')
	ax.set_ylabel('minibatch')
	cfg.savefig('{}{}'.format(cprof.outpath,'performance.pdf'),fig=fig)


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
		process_snapshot(processed,pfunc.fwithparams(weights),X,Y,i)		

	plotexample(unprocessed,processed)
	cfg.clearcurrenttask()
	return processed

def lplot():
	cprof=cfg.currentprofile()
	processandplot(cprof.unprocessed,cprof.learner,cprof.X_test,cprof.Y_test)


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
	cprof=cfg.currentprofile()
	figtitle=info(separator='\n')
	figpath='{}{} minibatches'.format(cprof.outpath,int(cprof.unprocessed.getval('minibatchnumber')))
	plotfunctions(cprof.sections,cprof.learner.eval,figtitle,figpath)


# dashboard
####################################################################################################

def act_on_input(key):
	if key=='q': quit()
	if key=='l': lplot()
	if key=='f': fplot()
	if key=='o':
		cfg.showfile(cfg.getoutpath())
	return key






def prepdashboard():
	cprof=cfg.currentprofile()
	import cdisplay

	# columndisplay

	instructions=cprof.instructions+'\n\n\nPress [l] (lowercase L) to generate learning plots.\n'+\
		'Press [f] to generate functions plot.\nPress [o] to open output folder.\
		\n\nPress [b] to break from current task.\nPress [q] to quit. '

	DB=cprof.dashboard
	w=DB.width; h=DB.height
	a,b,c,d=5,w//2-5,w//2+5,w-5
	y0=3

	# cdisplay1

	CD1=cdisplay.ConcreteDisplay(xlim=(a,b),ylim=(y0,h//2))
	CD1.add(disp.StaticText(msg=instructions))
	CD1.add(disp.VSpace(2))
	CD1.add(disp.Hline())
	CD1.add(disp.LogDisplay(height=10))
	CD1.add(disp.Hline())
	CD1.draw()

	CD2=cdisplay.ConcreteDisplay(xlim=(a,b),ylim=(h//2,3*h//4))
	statusdisplay=disp.StackedDisplay(memory=cprof.run)
	statusdisplay.add(disp.RunText(query='currenttask',msgtransform=lambda msg:msg if cfg.getcurrenttask()!=None else ''))
	statusdisplay.add(disp.Bar('currenttaskcompleteness',msgtransform=lambda msg:msg if cfg.getcurrenttask()!=None else ''))
	cprof.statusdisplay=statusdisplay
	CD2.add(statusdisplay)

	# infodisplay

	CD3=cdisplay.ConcreteDisplay(xlim=(c,d),ylim=(y0,3*h//4))
	CD3.add(disp.RunText(query='runinfo',wrap=True))


	cprof.run.addlistener(CD1,'recentlog')
	cprof.run.addlistener(CD2,'currenttaskcompleteness')
	cprof.run.addlistener(CD2,'target |Af|')
	cprof.run.addlistener(CD3,'runinfo')


	cprof.dashboard.add(CD1,'logcolumn')
	cprof.dashboard.add(CD2,'status')
	cprof.dashboard.add(CD3,'info')






def addlearningdisplay():
	import cdisplay
	cprof=cfg.currentprofile()

	DB=cprof.dashboard

	w=DB.width; h=DB.height
	a,b=5,w-5

	CD4=cdisplay.ConcreteDisplay((a,b),(h-10,h-1))
	CD4.add(disp.NumberPrint('minibatch loss',msg='training loss {:.3}'))
	CD4.add(disp.Bar('minibatch loss',style='.',emptystyle=' '))
	CD4.add(disp.Bar('minibatch loss',style=disp.BOX,avg_of=25))
	CD4.add(disp.VSpace(1))
	CD4.add(disp.NumberPrint('minibatchnumber',msg='minibatch number {:.0f}'))

	cprof.learningdisplay=CD4
	return cprof.dashboard.add(CD4)



