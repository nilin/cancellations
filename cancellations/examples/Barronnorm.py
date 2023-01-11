#
# nilin
# 
# 2022/7
#


import jax.numpy as jnp
from functools import partial
import jax
from jax.tree_util import tree_map

from cancellations.functions import _functions_
from cancellations.functions._functions_ import Product
from cancellations.lossesandnorms import losses,losses2
from cancellations.examples import harmonicoscillator2d

from cancellations.config.tracking import log
from cancellations.run import supervised


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################


def getlearner(profile):
    return Product(_functions_.IsoGaussian(1.0),\
        _functions_.ASBarron(n=profile.n,d=profile.d,m=100))
        #_functions_.ASBarron(**profile.learnerparams['ASNN']))

class Barronweight(losses.Lossgrad):
    def __init__(self,p,fd,density):
        self.p=p
        lossfn=lambda params, X, fX: self.loss(p,params)
        #self._eval_=jax.jit(jax.value_and_grad(lossfn))
        self._eval_=jax.jit(jax.value_and_grad(lossfn))

    @staticmethod
    def loss(p,prodparams):
        _,params=prodparams
        (W,b),a=params
        w1=jnp.squeeze(abs(a))
        w2=jnp.sum(jnp.abs(W),axis=(-2,-1))+jnp.abs(b)
        assert(w1.shape==w2.shape)
        if p==float('inf'):
            return jnp.max(w1*w2)
        else:
            return jnp.average((w1*w2)**p)**(1/p)

class ThresholdLG(losses.Lossgrad):
    def __init__(self,LG,delta,*a,**kw):
        self.delta=delta
        self.lg=LG(*a,**kw)

    def _eval_(self,params,X,Y):
        v,g=self.lg._eval_(params,X,Y)
        if v<self.delta: g=tree_map(lambda A:0*A,g)
        return v,g

class Run(supervised.Run):
    processname='Barron_norm'

    @classmethod
    def getdefaultprofile(cls):
        profile=super().getdefaultprofile().butwith(\
            getlearner=getlearner,\
            gettarget=harmonicoscillator2d.gettarget,\
            samples_train=10**5,\
            weight_decay=0.0)

        profile.initlossgrads=[partial(ThresholdLG,losses.Lossgrad_SI,.001),partial(Barronweight,1.0)]
        profile.lossnames=['eps','Barron norm estimate']
        profile.lossweights=[100.0,1.0]

        return profile
