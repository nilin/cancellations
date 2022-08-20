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
from functions import FunctionDescription as FD

jax.config.update("jax_enable_x64", True)






cfg.exname='ASNN_learn_slater'

cfg.explanation='Example '+cfg.exname

cfg.params={
'n':5,
'd':2,
'learnertype':'AS_NN',
'learnerwidths':['d_sp',100,1],
'spwidths':['d',100,100],
'targettype':'hermiteSlater',
'weight_decay':0,
'lossfn':'SI_loss',
'samples_train':25000,
'samples_test':1000,
'iterations':100000,
'minibatchsize':64
}

instructions='instructions:\n\npython e_{}.py (t/lr) \n\n\
parameters represent:\ntanh/leaky relu\n'.format(cfg.exname)


try:
	examples.adjustparams(learneractivation=getfromargs(t='tanh',lr='leakyrelu'))
except:
	db.clear()
	print(instructions)
	quit()
globals().update(cfg.params)


target=functions.get_func(ftype=targettype,n=n,d=d)
learner=functions.get_composed_func(FD('singleparticleNN',d=d,widths=spwidths,activation=learneractivation),FD(learnertype,n=n,d=100,widths=learnerwidths,activation=learneractivation))
examples.runexample0(target,learner)
