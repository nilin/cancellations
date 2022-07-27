# nilin


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

jax.config.update("jax_enable_x64", True)






explanation='\
In order not to give an unfair advantage to either activation function \n\
we let the target function in this example be the sum of two antisymmetrized NNs, \n\
one for each activation function. Both NNs are normalized to have the same magnitude.'







def saveplots(plotpath,learned,target):

	plt.close('all')
	fig1,ax1=plt.subplots(1)
	fig2,(ax21,ax22)=plt.subplots(1,2)

	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)
	pt.plotalongline(ax1,target,learned,X_test,fineness=1000)
	pt.ploterrorhist(ax21,cfg.gethists())
	pt.ploterrorhist(ax22,cfg.gethists(),logscale=True)

	cfg.savefig('{}/{} {}'.format(plotpath,int(cfg.timestamp()),'s.pdf'),fig1)
	cfg.savefig(plotpath+'/losses.pdf',fig2)
	
	return fig1,fig2



def test(learned,target):
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)
	Y_test=target(X_test)
	return cfg.lossfn(learned(X_test),Y_test)


def run():

	params={
	'targettype':'AS_NN',
	'learnertype':'AS_NN',
	'n':5,
	'd':1,
	'samples_train':10000,
	'samples_val':100,
	'samples_test':1000,
	'targetwidths':[5,25,25,25,1],
	'learnerwidths':[5,250,1],
	#'targetactivation':'tanh',
	'learneractivation':'ReLU',
	'checkpoint_interval':5,
	'timebound':600
	}


	redefs=cfg.get_cmdln_args()
	globals().update(params)
	globals().update(redefs)
	varnames=cfg.orderedunion(params,redefs)


	ignore={'plotfineness','minibatchsize','initfromfile','samples_test','samples_val','d','checkpoint_interval'}
	#if 'NN' not in targettype: ignore.update({'targetwidths','targetactivation'})
	assert('NN' in targettype)

	sessioninfo=cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
	cfg.setval('sessioninfo',sessioninfo)
	cfg.log('sessioninfo:\n'+sessioninfo)
	plotpath='plots/'+cfg.sessionID()
	cfg.savetxt(plotpath+'/info.txt',sessioninfo)





	cfg.log('Generating AS functions.')

	targets=dict()
	for ac in activations.keys():
		t_args=(n,d,targetwidths,activations[ac])
		#if targettype=='HermiteSlater': t_args=(n,'H',1/8)
		targets[ac]={\
		'AS_NN':ASf.init_static(ASf.init_AS_NN),\
		'SlaterSumNN_singlePhi':ASf.init_static(ASf.init_SlaterSumNN_singlePhi),\
		'SlaterSumNN_nPhis':ASf.init_static(ASf.init_SlaterSumNN_nPhis),\
		'HermiteSlater':examplefunctions.HermiteSlater\
		}[targettype](*t_args)

	l_args=(n,d,learnerwidths,activations[learneractivation])
	learner={\
	'AS_NN':ASf.init_AS_NN,\
	'SlaterSumNN_singlePhi':ASf.init_SlaterSumNN_singlePhi,\
	'SlaterSumNN_nPhis':ASf.init_SlaterSumNN_nPhis\
	}[learnertype](*l_args)

	cfg.log('Generating training data X.')
	X=rnd.uniform(cfg.nextkey(),(samples_train+samples_val,n,d),minval=-1,maxval=1)

	cfg.log('normalizing targets')
	for ac in activations.keys():
		targets[ac]=util.normalize(targets[ac],X[:100])

	target=jax.jit(lambda X:targets['ReLU'](X)+targets['tanh'](X))


	cfg.log('Verifying antisymmetry of target.')
	testing.verify_antisymmetric(target,n,d)

	cfg.log('Verifying antisymmetry of learner.')
	testing.verify_antisymmetric(util.fixparams(*learner),n,d)

	cfg.log('Generating training data Y.')
	Y=target(X)


	#----------------------------------------------------------------------------------------------------
	# train
	#----------------------------------------------------------------------------------------------------


	trainer=learning.Trainer(learner,X,Y)

	stopwatch=cfg.Stopwatch()
	while time.perf_counter()<timebound:
		if stopwatch.elapsed()<checkpoint_interval:
			trainer.step()
			cfg.pokelisteners('refresh')
		else:
			stopwatch.tick()

			trainer.save()
			learned=trainer.get_learned()
			cfg.setval('test loss',test(learned,target))
			saveplots(plotpath,learned,target)
			cfg.log('evaluated test loss')	



if __name__=='__main__':

	slate=db.Slate()

	slate.addtext(explanation,height=4)
	slate.addtext(lambda tk:tk.get('sessioninfo'),height=10)
	slate.addtext(lambda tk:[s+50*' ' for s in tk.gethist('log')[-20:]],height=20)
	slate.addspace(2)
	slate.addtext('training loss of 10, 100 minibatches')
	slate.addbar(lambda tk:np.average(np.array(tk.gethist('minibatch loss'))[-10:]),emptystyle='.')
	slate.addbar(lambda tk:np.average(np.array(tk.gethist('minibatch loss'))[-100:]),emptystyle='.')
	slate.addspace(2)
	slate.addtext('test loss')
	slate.addbar(lambda tk:tk.get('test loss'),emptystyle='.')

	run()
