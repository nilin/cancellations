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
from config import session
import testing
import functions



def getrunfn0(target,learner):

	def runfn():
		globals().update(cfg.params)

		global unprocessed,X,X_test,Y,Y_test,sections,_learner_,_target_
		_target_,_learner_=target,learner
		
		sessioninfo='{}\nsessionID: {}\n\n{}'.format(cfg.explanation,cfg.sessionID,INFO())
		session.remember('sessioninfo',sessioninfo)
		cfg.write(session.getval('sessioninfo'),cfg.outpath+'info.txt',mode='w')




		unprocessed=cfg.ActiveMemory()

		try:
			cfg.dashboard.add_display(Display2(10,cfg.dashboard.width,unprocessed),2*cfg.dashboard.height//3,name='bars')
		except:
			pass



		cfg.currentkeychain=2
		X=cfg.genX(samples_train)
		X_test=cfg.genX(samples_test)

		cfg.logcurrenttask('preparing training data')
		Y=target.eval(X)
		cfg.logcurrenttask('preparing test data')
		Y_test=target.eval(X_test)

		setupdata={k:globals()[k] for k in ['X_test','Y_test']}|{'target':target.compress(),'learner':learner.compress()}
		cfg.save(setupdata,cfg.outpath+'data/setup')

		trainer=learning.Trainer(learner,X,Y,weight_decay=weight_decay,minibatchsize=minibatchsize,lossfn=util.SI_loss) #,lossgrad=mv.gen_lossgrad(AS,lossfn=util.SI_loss))
		lazyplot=cfg.Clockedworker()
		cfg.logcurrenttask('preparing slices for plotting')
		cfg.currentkeychain=4
		sections=pt.genCrossSections(X,Y,target.eval)

		cfg.regsched=cfg.Scheduler(cfg.nonsparsesched(iterations,start=100))
		cfg.provide(plotsched=cfg.Scheduler(cfg.sparsesched(iterations,start=1000)))

		cfg.logcurrenttask('begin training')
		for i in range(iterations+1):

			cfg.poke()
			loss=trainer.step()

			unprocessed.addcontext('minibatchnumber',i)
			unprocessed.remember('minibatch loss',loss)

			if cfg.regsched.activate(i):
				unprocessed.remember('weights',learner.weights)
				cfg.save(unprocessed,cfg.outpath+'data/unprocessed')

			if cfg.plotsched.activate(i):
				fplot()
				lazyplot.do_if_rested(.2,lplot)
	return runfn


def testantisymmetry(target,learner,X):
	try:
		cfg.logcurrenttask('verifying antisymmetry of target')
		testing.verify_antisymmetric(target.eval,X[:100])
		cfg.logcurrenttask('verifying antisymmetry of learner')
		testing.verify_antisymmetric(learner.eval,X[:100])
		return True
	except:
		cfg.log('Warning: not antisymmetric')
		return False


def adjustnorms(Afdescr,X,iterations=100):
	Af=Afdescr.f
	f=functions.switchtype(Afdescr).f
	weights=Afdescr.weights

	normratio=jax.jit(lambda weights,X:util.norm(f(weights,X))/util.norm(Af(weights,X)))
	cfg.log('|f|/|Af|={:.3} before'.format(normratio(weights,X)))

	@jax.jit
	def directloss(params,X):
		return util.ReLU(-jnp.log(util.norm(Af(params,X))))+jnp.log(util.norm(f(params,X)))

	trainer=learning.DirectlossTrainer(directloss,weights,X)
	for i in range(iterations):
		cfg.trackcurrenttask('adjusting target norm',i/iterations)
		trainer.step()

	weights=trainer.learner.weights
	cfg.log('|f|/|Af|={:.3} after'.format(normratio(weights,X)))
	return weights


def lplot():
	processandplot(unprocessed,_learner_,X_test,Y_test)
def fplot():
	figtitle=info()
	figpath='{}{} minibatches'.format(cfg.outpath,int(unprocessed.getval('minibatchnumber')))
	plotfunctions(sections,_learner_.eval,figtitle,figpath)

def info(separator=' | '):
	return 'n={}, target: {}{}learner: {}'.format(n,_target_.richtypename(),separator,_learner_.richtypename())

def INFO(separator='\n\n'):
	lb='\n'+50*db.dash+'\n'
	targetinfo='target\n\n{}'.format(cfg.indent(_target_.getinfo()))
	learnerinfo='learner\n\n{}'.format(cfg.indent(_learner_.getinfo()))
	return lb+'n={}'.format(n)+lb+targetinfo+'\n'*4+learnerinfo+lb

def process_input(c):
	if c==108: lplot()
	if c==102: fplot()




# learning plots
####################################################################################################


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

		cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))
		process_snapshot(processed,pfunc.fwithparams(weights),X,Y,i)		

	plotexample(unprocessed,processed)
	#cfg.save(processed,cfg.outpath+'data/processed')

	cfg.clearcurrenttask()
	return processed






# function plots
####################################################################################################


def plotfunctions(sections,f,figtitle,path):
	cfg.logcurrenttask('generating function plots')
	plt.close('all')
	for fignum,section in enumerate(sections):
		fig=section.plot_y_vs_f_SI(f)
		cfg.trackcurrenttask('generating functions plots',(fignum+1)/len(sections))
		fig.suptitle(figtitle)
		cfg.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
	cfg.clearcurrenttask()




# prep
####################################################################################################


def adjustparams(**priorityparams):

	cfg.params=priorityparams|cfg.params
	params=cfg.params
	explanation=cfg.explanation
	params.update(cfg.cmdredefs)




class Display2(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		self.addnumberprint('minibatch loss',msg='training loss {:.3}')
		self.addbar('minibatch loss',style='.')
		self.addbar('minibatch loss',style=db.BOX,avg_of=100)
		self.addspace()
		self.addline()
		self.addnumberprint('minibatchnumber',msg='minibatch number {:.0f}')


def pickdisplay():
	try:
		return cfg.selectonefromargs('nodisplay','logdisplay')
	except:
		return 'fulldisplay'
	

def runexample(runfn):
	cfg.trackduration=True
	if 'debug' in cfg.cmdparams:
		import debug
		displaymode='logdisplay'
	else:
		displaymode=pickdisplay()

	db.clear()
	if displaymode=='fulldisplay':
		import run_in_display
		run_in_display.RID(runfn,process_input)
	else:
		runfn()


def runexample0(target,learner):
	runexample(getrunfn0(target,learner))


