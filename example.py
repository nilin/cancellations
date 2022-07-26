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
from AS_functions import init_AS_NN,init_SlaterSumNN,gen_static_AS_NN,gen_static_SlaterSumNN






def make_dashboard(TK):
	D=db.Dashboard(TK)
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

	pt.plotalongline(ax1,target,learned,X_test,fineness=plotfineness)
	pt.ploterrorhist(ax21,'data/hist')
	pt.ploterrorhist(ax22,'data/hist',logscale=True)

	cfg.savefig('{}/{} {}'.format(plotpath,int(TK.timestamp()),'s.pdf'),fig1)
	cfg.savefig(plotpath+'/losses.pdf',fig2)
	
	return fig1,fig2



TK=cfg.HistTracker()
tk=cfg.Tracker()
D=make_dashboard(TK)



params={
'learnertype':'AS_NN',
'targettype':'SlaterSumNN',
'n':5,
'd':1,
'samples_train':10000,
'samples_val':100,
'samples_test':1000,
'learnerwidths':[25,100],
'targetwidths':[25,25,25],
'plotfineness':1000,
'checkpoint_interval':5
}


redefs=cfg.get_cmdln_args()
globals().update(params)
globals().update(redefs)
varnames=cfg.orderedunion(params,redefs)


ignore={'plotfineness','minibatchsize','initfromfile','samples_test','samples_val','d','checkpoint_interval'}
sessioninfo=cfg.formatvars([(k,globals()[k]) for k in varnames],separator='\n',ignore=ignore)
TK.set('sessioninfo',sessioninfo)
plotpath='plots/'+TK.ID
cfg.savetxt(plotpath+'/info.txt',sessioninfo)





TK.log('Generating AS functions.')
t_args=(n,d,targetwidths)
l_args=(n,d,learnerwidths)
target={'AS_NN':gen_static_AS_NN(*t_args),'SlaterSumNN':gen_static_SlaterSumNN(*t_args)}[learnertype]
learner={'AS_NN':init_AS_NN(*l_args),'SlaterSumNN':init_SlaterSumNN(*l_args)}[learnertype]



TK.log('Generating training data X.')
X=rnd.uniform(cfg.nextkey(),(samples_train+samples_val,n,d),minval=-1,maxval=1)


TK.log('Generating training data Y.')
target=util.normalize(target,X[:100])
Y=target(X)
TK.log('Training data done.')

learner=AS_functions.init_AS_NN(n,d,learnerwidths)
Af,_,_=learner



#----------------------------------------------------------------------------------------------------
# train
#----------------------------------------------------------------------------------------------------


trainer=learning.TrainerWithValidation(learner,X,Y,tracker=TK,validationbatchsize=samples_val)
db.clear()

stopwatch=cfg.Stopwatch()
while True:
	if stopwatch.elapsed()<checkpoint_interval:
		trainer.step()
		tk.set('stopwatch',stopwatch.elapsed())
		D.refresh()
	else:
		stopwatch.tick()
		trainer.validation()
		trainer.save()
		try:
			saveplots(Af,target)
		except Exception as e:
			TK.log(str(e))


