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
from cancellations.examples import examples

from cancellations.config.tracking import log
from cancellations.run import supervised,runtemplate
from cancellations.config.tracking import Profile


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################


def getBarronfn(profile):
    return Product(_functions_.IsoGaussian(1.0),\
        _functions_.ASBarron(n=profile.n,d=profile.d,m=profile.m))
        #_functions_.ASBarron(**profile.learnerparams['ASNN']))


class Run(runtemplate.Run):
    processname='Barron_norm'

    @classmethod
    def getdefaultprofile(cls,**kwargs):
        P=profile=super().getdefaultprofile(**kwargs)
        Barron=getBarronfn(profile)
        profile.learner=Barron
        profile.target=runtemplate.getlearner_example(Profile(n=P.n,d=P.d,ndets=P.mtarget))
        #profile.target=examples.get_harmonic_oscillator2d(profile)
        profile.lossgrads=[\
            get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_,profile),.001),\
            get_barronweight(1.0,Barron._eval_,profile),\
            #losses.get_lossgrad_normalization(profile.learner._eval_,profile)
        ]
        profile.lossnames=['eps','Barron norm estimate']
        profile.lossweights=[100.0,0.1]
        profile.sampler=profile.getXYsampler(profile.Xsampler,profile.target)
        return profile

    @classmethod
    def getprofiles(cls):
        return {\
            'm=1': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=1),
            'm=2': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=2),
            'm=4': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=4),
            'm=8': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=8),\
        }



def get_barronweight(p,Barronfn,profile):
    norm=partial(losses.norm,profile.X_density,Barronfn)
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
    lossfn=lambda params,X,*Y: loss(p,params)/norm(params,X)
    return jax.jit(jax.value_and_grad(lossfn))

def get_threshold_lg(lg,delta):
    def _eval_(params,*X):
        val,grad=lg(params,*X)
        weight=jax.nn.sigmoid(val/delta-1)
        return val,tree_map(lambda A:weight*A,grad)
    return jax.jit(_eval_)
