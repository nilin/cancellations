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
import pdb
import time
import math
import dashboard as db
import time
import AS_tools
import examplefunctions
import AS_functions as ASf






def make_dashboard():
	D=db.Dashboard()
	D.addtext(lambda tk:tk.get('sessioninfo'))
	D.addspace(10)
	D.addlog(10)
	#D.addtext('time to next validation set/save')
	#D.addbar((lambda tk:1-tk.get('stopwatch')/checkpoint_interval),style=db.dash,tracker=tk)
	#D.addspace(1)
	D.addtext(lambda tk:'{:,} samples left in epoch'.format(tk.get('minibatches left')*tk.get('minibatchsize')))
	D.addbar(lambda tk:tk.get('minibatches left')/tk.get('minibatches'),style=db.dash)
	D.addspace(5)
	D.addtext('training loss of last minibatch, 100 minibatches')
	D.addbar(lambda tk:tk.get('minibatch loss'),style=db.dash)
	#D.addbar(lambda tk:np.average(np.array(tk.gethist('minibatch loss'))[-10:]),style=db.box)
	D.addbar(lambda tk:np.average(np.array(tk.gethist('minibatch loss'))[-100:]))
	D.addspace()
	D.addtext('validation loss')
	D.addbar(lambda tk:tk.get('validation loss'))
	return D


def saveplots(Af,target):

	plt.close('all')

	learned=AS_functions.Af_from_hist('data/hist',Af)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)

	fig1,ax1=plt.subplots(1)
	fig2,(ax21,ax22)=plt.subplots(1,2)

	pt.plotalongline(ax1,target,learned,X_test,fineness=1000)
	pt.ploterrorhist(ax21,'data/hist')
	pt.ploterrorhist(ax22,'data/hist',logscale=True)

	cfg.savefig('{}/{} {}'.format(plotpath,int(cfg.timestamp()),'s.pdf'),fig1)
	cfg.savefig(plotpath+'/losses.pdf',fig2)
	
	return fig1,fig2







D=make_dashboard()

params={
'learnertype':'AS_NN',
'targettype':'SlaterSumNN_nPhis',
'n':5,
'd':1,
'samples_train':10000,
'samples_val':100,
'samples_test':1000,
'learnerwidths':[25,25,25],
'targetwidths':[25,25,25],
'checkpoint_interval':5
}


redefs=cfg.get_cmdln_args()
globals().update(params)
globals().update(redefs)
varnames=cfg.orderedunion(params,redefs)


ignore={'plotfineness','minibatchsize','initfromfile','samples_test','samples_val','d','checkpoint_interval'}
sessioninfo=cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
cfg.setval('sessioninfo',sessioninfo)
plotpath='plots/'+cfg.sessionID()
cfg.savetxt(plotpath+'/info.txt',sessioninfo)





cfg.log('Generating AS functions.')
t_args=(n,d,targetwidths)
l_args=(n,d,learnerwidths)

target={\
'AS_NN':ASf.gen_static_AS_NN(*t_args),\
'SlaterSumNN_singlePhi':ASf.gen_static_SlaterSumNN_singlePhi(*t_args),\
'SlaterSumNN_nPhis':ASf.gen_static_SlaterSumNN_nPhis(*t_args),\
'HermiteSlater':examplefunctions.HermiteSlater(n,'H',1/8)\
}[targettype]

learner={\
'AS_NN':ASf.init_AS_NN,\
'SlaterSumNNsinglePhi':ASf.init_SlaterSumNN,\
'SlaterSumNN_nPhis':ASf.init_SlaterSumNN_nPhis\
}[learnertype](*l_args)



cfg.log('Generating training data X.')
X=rnd.uniform(cfg.nextkey(),(samples_train+samples_val,n,d),minval=-1,maxval=1)


cfg.log('Generating training data Y.')
target=util.normalize(target,X[:100])
Y=target(X)
cfg.log('Training data done.')

learner=AS_functions.init_AS_NN(n,d,learnerwidths)
Af,_,_=learner



#----------------------------------------------------------------------------------------------------
# train
#----------------------------------------------------------------------------------------------------


trainer=learning.TrainerWithValidation(learner,X,Y,validationbatchsize=samples_val)
db.clear()

stopwatch=cfg.Stopwatch()
while True:
	if stopwatch.elapsed()<checkpoint_interval:
		trainer.step()
		cfg.set_temp('stopwatch',stopwatch.elapsed())
		D.refresh()
	else:
		stopwatch.tick()
		trainer.validation()
		trainer.save()
		try:
			saveplots(Af,target)
		except Exception as e:
			cfg.log(str(e))


