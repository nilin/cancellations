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
jax.config.update("jax_enable_x64", True)






cfg.exname='example'
cfg.explanation=''
cfg.outpath='outputs/{}/{}/'.format(cfg.exname,cfg.sessionID)

cfg.log('imports done')
#try:
#	learneractivation=getfromargs(t='tanh',lr='leakyrelu')
#except Exception as e:
#	db.clear()
#	print(instructions)
#	print(str(e))
#	quit()

n=5
d=2


####################################################################################################

#target=ComposedFunction(functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen'),'tanh')
#target=ComposedFunction(functions.Slater('parallelgaussians',n=n,d=d,mode='gen'),'tanh')
target=ComposedFunction(functions.ASNN(n=n,d=d,widths=['nd',10,10,1],activation='tanh'),'tanh')

cfg.log('target prepared')



####################################################################################################
learneractivation='tanh'
#learneractivation='leakyrelu'

#learner=functions.Slater(SingleparticleNN(widths=[d,10,10,n],activation=learneractivation))
#
#d_=10;
#learner=ComposedFunction(\
#	SingleparticleNN(widths=[d,100,100,d_],activation='tanh'),\
#	functions.ASNN(n=n,d=d_,widths=['nd',100,1],activation=learneractivation))

d_=25; k=25;
learner=ComposedFunction(\
SingleparticleNN(widths=[2,100,d_],activation=learneractivation),\
#functions.Backflow(activation=learneractivation,widths=[d_,d_]),\
functions.Wrappedfunction('detsum',n=n,d=d_,ndets=k)\
)

cfg.log('learner prepared')
####################################################################################################


cfg.addparams(
weight_decay=0,
lossfn='SI_loss',
samples_train=25000,
samples_test=1000,
iterations=10000,
minibatchsize=None
)
globals().update(cfg.params)
cfg.register(globals(),['n','d'])


exampletemplate.runexample0(target,learner)
