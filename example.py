#
# nilin
# 
# 2022/7
#


from re import I
import config as cfg
import functions
import dashboard as db
from config import session
import exampletemplate
import jax
from functions import ComposedFunction,SingleparticleNN
import jax.random as rnd
import browse_runs
import os
jax.config.update("jax_enable_x64", True)






cfg.exname='example'
cfg.explanation=''
cfg.outpath='outputs/{}/{}/'.format(cfg.exname,cfg.sessionID)

cfg.log('imports done')

n=5
d=2

cfg.X_distr=lambda key,samples:rnd.uniform(key,(samples,n,d),minval=-1,maxval=1)

####################################################################################################

if 'loadtarget' in cfg.cmdparams:
    path=browse_runs.pickfolders(multiple=False,msg='Choose target from previous run.',\
        condition=lambda path:os.path.exists(path+'/data/setup'))+'data/setup'
    target=cfg.load(path)['target']
    target.restore()
else:
    #target=ComposedFunction(functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen'),'tanh')
    #target=ComposedFunction(functions.Slater('parallelgaussians',n=n,d=d,mode='gen'),'tanh')
    target=ComposedFunction(functions.ASNN(n=n,d=d,widths=['nd',10,10,1],activation='tanh'),'tanh')


    exampletemplate.adjustnorms(target,X=cfg.genX(10000))
    cfg.log('target initialized')

####################################################################################################
learneractivation='tanh'
#learneractivation='leakyrelu'

#learner=functions.Slater(SingleparticleNN(widths=[d,10,10,n],activation=learneractivation))
#
#d_=10;
#learner=ComposedFunction(\
#	SingleparticleNN(widths=[d,100,100,d_],activation='tanh'),\
#	functions.ASNN(n=n,d=d_,widths=['nd',100,1],activation=learneractivation))

d_=25; ndets=25;
learner=ComposedFunction(\
SingleparticleNN(widths=[d,100,d_],activation=learneractivation),\
functions.Backflow(activation=learneractivation,widths=[d_,d_]),\
functions.DetSum(n=n,d=d_,ndets=ndets)
)
cfg.log('learner initialized')
####################################################################################################

exampletemplate.testantisymmetry(target,learner,X=cfg.genX(100))

####################################################################################################

cfg.addparams(
weight_decay=0,
lossfn='SI_loss',
samples_train=100000,
samples_test=1000,
iterations=10000,
minibatchsize=None
)
cfg.register(globals(),['n','d'])


exampletemplate.runexample0(target,learner)
