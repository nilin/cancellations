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






cfg.exname='test'

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

#learner=functions.Slater(SingleparticleNN(widths=[2,100,100,5],activation=learneractivation))
#learner=ComposedFunction(SingleparticleNN(widths=[2,100,100,20],activation=learneractivation),functions.ASNN(n=n,d=20,widths=['nd',100,1],activation=learneractivation))
learner=ComposedFunction(SingleparticleNN(widths=[2,100,100,20],activation='tanh'),functions.ASNN(n=n,d=20,widths=['nd',100,1],activation=learneractivation))
#learner=functions.Slater(SingleparticleNN(widths=[2,10,10,5],activation=learneractivation))
#learner=ComposedFunction(SingleparticleNN(widths=[2,100,100],activation=learneractivation),functions.BackflowAS(n=n,widths=[100,100],k=5,activation=learneractivation))



#target=functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen')
#target=functions.Slater('parallelgaussians',n=n,d=d,mode='gen')
#target=functions.Slater(SingleparticleNN(widths=[2,10,10,5],activation='tanh'))
target=ComposedFunction(SingleparticleNN(widths=[2,10,10],activation='tanh'),functions.ASNN(n=n,d=10,widths=['nd',10,1],activation='tanh'))

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
