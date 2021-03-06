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
import examplefunctions
import AS_functions as ASf

import e1

#jax.config.update("jax_enable_x64", True)




# e2
explanation='Example 2\n\
In order not to give an unfair advantage to either activation function \n\
we let the target function in this example be the sum of two antisymmetrized NNs, \n\
one constructed with each activation function. Both NNs are normalized to have the same magnitude.'


exname='e2'


def run(cmdargs):

	params={
	'targettype':'AS_NN',
	'learnertype':'AS_NN',
	'n':5,
	'd':1,
	'samples_train':10000,
	'samples_test':250,
	'targetwidths':[5,5,5,5,5,1],
	'learnerwidths':[5,250,1],
	# e2
	#'targetactivation':both,
	#'learneractivation':?,
	'checkpoint_interval':2.5,
	'timebound':120
	}
	args,redefs=cfg.parse_cmdln_args(cmdargs)



	# e2
	try:
		l_a={'r':'ReLU','relu':'ReLU','ReLU':'ReLU','t':'tanh','tanh':'tanh'}[args[0]]
	except:
		print(10*'\n'+'Pass activation function as first parameter.\n'+db.wideline()+10*'\n')	
		sys.exit(0)

	params['learneractivation']=l_a
	

	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)
	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','d','checkpoint_interval'}







	# e2
	cfg.outpaths.add('outputs/{}/{}/{}/'.format(exname,learneractivation,cfg.sessionID))
	cfg.outpaths.add('outputs/{}/{}/lastrun/'.format(exname,learneractivation))


	sessioninfo=explanation+'\n\n'+cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setstatic('sessioninfo',sessioninfo)
	cfg.log('sessioninfo:\n'+sessioninfo)
	cfg.write(sessioninfo,'outputs/{}/info.txt'.format(exname),mode='w')


	#----------------------------------------------------------------------------------------------------
	cfg.log('Generating AS functions.')

	# e2
	targets=[ASf.init_target(targettype,n,d,targetwidths,activations[ac]) for ac in ['ReLU','tanh']]
	learner=ASf.init_learner(learnertype,n,d,learnerwidths,activations[learneractivation])

	
	#----------------------------------------------------------------------------------------------------
	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	# e2
	cfg.log('normalizing target terms')
	targets=[util.normalize(target,X[:100]) for target in targets]
	target=jax.jit(lambda X:targets[0](X)+targets[1](X))

	cfg.log('\nVerifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(learner.as_static(),n,d)

	cfg.log('\nGenerating data Y.')
	Y=target(X)
	Y_test=target(X_test)




	cfg.log('Preparing cross sections for plotting.')
	sections=pt.CrossSections(X,Y,target,3)




	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------




	# e2
	slate.addspace(2)
	#slate.addtext(lambda *_:'magnitudes of weights in each layer: {}'.format(cfg.terse([util.norm(W) for W in cfg.getval('weights')[0]])))
	#slate.addtext(lambda *_:'||f|| = {:.2f}'.format(cfg.getval('NS norm')))
	#slate.addtext(lambda *_:'||f||/||Af|| = {:.2f}'.format(cfg.getval('normratio')))


	trainer=learning.Trainer(learner,X,Y)
	sc1=cfg.Scheduler(cfg.arange(60,3600,60))
	sc3=cfg.Scheduler(cfg.expsched(.1,.1))
	cfg.log('\nStart training.\n')

	while sc1.elapsed()<timebound:
		trainer.step()
		cfg.pokelisteners('refresh')

		if sc1.dispatch():
			trainer.save()
			fig1=e1.getfnplot(sections,trainer.get_learned())
			fig2=e1.getlossplots()
			fig3,fig4=e1.getnormplots()

			for fig in [fig1,fig2,fig3,fig4]:
				fig.title(learneractivation)
			cfg.savefig(*['{}{}{}'.format(path,int(sc1.elapsed()),'s.pdf') for path in cfg.outpaths],fig=fig1)
			cfg.savefig(*[path+'losses.pdf' for path in cfg.outpaths],fig=fig2)
			cfg.savefig(*[path+'fnorm.pdf' for path in cfg.outpaths],fig=fig3)
			cfg.savefig(*[path+'Afnorm.pdf' for path in cfg.outpaths],fig=fig4)

			# e2
			try:
				fig3=getlosscomparisonplots({ac:'outputs/{}/{}/lastrun/hist'.format(exname,ac) for ac in activations})
				cfg.savefig('outputs/{}/comparetraining.pdf'.format(exname),fig=fig3)
			except Exception as e:
				cfg.log('Comparison plot of losses (outputs/[examplename]/comparetraining.pdf) will be generated once script has run with both activation functions.')



		if sc3.dispatch():
			cfg.trackhist('test loss',cfg.lossfn(learner.as_static()(X_test),Y_test))

			cfg.trackhist('NS norm',util.norm(learner.static_NS()(X_test)))
			cfg.trackhist('AS norm',util.norm(learner.as_static()(X_test)))

		



#----------------------------------------------------------------------------------------------------


# e2
def getlosscomparisonplots(histpaths):

	plt.close('all')
	fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))
	hists={ac:cfg.retrieve(histpaths[ac]) for ac in activations}

	plottestloss(ax21,hists)
	ax.set_ylim(0,1)
	plottestloss(ax22,hists,logscale=True)
	ax.set_yscale('log')

	return fig2

# e2
def plottestloss(ax,hists,logscale=False):

	ax.plot(*[hists['ReLU']['test loss'][_] for _ in ['timestamps','vals']],'bo-',label='ReLU')
	ax.plot(*[hists['tanh']['test loss'][_] for _ in ['timestamps','vals']],'rd:',label='tanh')
	
	ax.legend()
	ax.set_xlabel('seconds')

	if logscale:
	else:





# e2
def getnormcomparisonplots(histpaths):

	plt.close('all')
	fig2,(ax21,ax22)=plt.subplots(1,2,figsize=(15,7))
	hists={ac:cfg.retrieve(histpaths[ac]) for ac in activations}

	plottestloss(ax21,hists)
	plottestloss(ax22,hists,logscale=True)

	return fig2

# e2
def plottestloss(ax,hists,logscale=False):

	ax.plot(*[hists['ReLU']['test loss'][_] for _ in ['timestamps','vals']],'bo-',label='ReLU')
	ax.plot(*[hists['tanh']['test loss'][_] for _ in ['timestamps','vals']],'rd:',label='tanh')
	
	ax.legend()
	ax.set_xlabel('seconds')

	if logscale:
		ax.set_yscale('log')
	else:
		ax.set_ylim(0,1)


#----------------------------------------------------------------------------------------------------



if __name__=='__main__':

	slate=db.display_1()
	cfg.setstatic('display',slate)


	run(sys.argv[1:])
