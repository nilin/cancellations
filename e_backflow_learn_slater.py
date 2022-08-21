#
# nilin
# 
# 2022/7
#


import config as cfg
import functions
import dashboard as db
from config import session,getfromargs
import examples
import jax
from functions import getfunc,ComposedFunction,SingleparticleNN
import examplefunctions

jax.config.update("jax_enable_x64", True)






cfg.exname='backflow_learn_slater'

cfg.explanation='Example '+cfg.exname

instructions='instructions:\n\npython e_backflow_learn_ASNN.py (t/lr) \n\n\
parameters represent:\ntanh/leaky relu\n'

cfg.outpath='outputs/{}/{}/'.format(cfg.exname,cfg.sessionID)





try:
	learneractivation=getfromargs(t='tanh',lr='leakyrelu')
except Exception as e:
	db.clear()
	print(instructions)
	print(str(e))
	quit()
globals().update(cfg.params)


n=5
d=2
spwidths=[2,100,100]
bfwidths=[100,100]
learner=ComposedFunction(SingleparticleNN(widths=spwidths,activation=learneractivation),functions.BackflowAS(n=n,widths=bfwidths,k=5,activation=learneractivation))
#target=functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen')
target=functions.Slater('parallelgaussians',n=n,d=d,mode='gen')

cfg.register(globals(),['n','d'])
cfg.params.update({
'weight_decay':0,
'lossfn':'SI_loss',
'samples_train':25000,
'samples_test':1000,
'iterations':100000,
'minibatchsize':None
})

examples.runexample0(target,learner)
