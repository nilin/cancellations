#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses, Barronnorm
import jax.numpy as jnp
from functools import partial
import jax
from cancellations.config import tracking
from jax.tree_util import tree_map
from cancellations.display import _display_

from cancellations.functions import _functions_, examplefunctions as examples, examplefunctions
from cancellations.functions._functions_ import Product

import math
from cancellations.config import config as cfg
from cancellations.run import runtemplate
from cancellations.config.tracking import Profile, log, sysutil
import matplotlib.pyplot as plt
import re
import os


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################



class Run(Barronnorm.Run):
    processname='expnorm'
    Ansatzoptions=('envelope_on','envelope_off')

    @classmethod
    def get_Ansatz(cls,P):
        match P.mode:
            case 'envelope_on':
                return Product(_functions_.IsoGaussian(1.0),\
                    examplefunctions.ExpSlater(n=P.n,d=P.d,m=P.m))
            case 'envelope_off':
                return examplefunctions.ExpSlater(n=P.n,d=P.d,m=P.m)

    @classmethod
    def get_Ansatzweightnorm(cls,p,f):
        def loss(p,prodparams):
            _,params=prodparams
            (W,b),a=params
            w1=jnp.squeeze(abs(a))
            w2=jnp.sum(jnp.abs(W),axis=(-2,-1))+jnp.abs(b)
            assert(w1.shape==w2.shape)
            if p==float('inf'):
                return jnp.max(w1*w2)
            else:
                return jnp.sum((w1*w2)**p)**(1/p)
        lossfn=lambda params,X,Y,rho: loss(p,params)/losses.norm(f(params,X),rho)
        return jax.jit(jax.value_and_grad(lossfn))


### batch runs ###

class Runthrough(_display_.Process):
    def execprocess(self):
        P=self.profile
        imax=max(P.ns)
        for n in P.ns:
            P=Barronnorm.Run.getprofile(self,n=n,d=P.d,m=1024,mode='ANTISYM',imax=imax).butwith(iterations=10**4)
            self.subprocess(Run(profile=P))

            P=Run.getprofile(self,n=n,d=P.d,m=1024,mode='envelope_on',imax=imax).butwith(iterations=10**4)
            self.subprocess(Run(profile=P))

    @classmethod
    def getprofile(cls,parentprocess):
        P=tracking.Profile()
        P.d=parentprocess.browse(options=[1,2,3],msg='Pick d')
        P.ns=parentprocess.browse(options=[1,2,3,4,5,6],onlyone=False,msg='Pick ns')
        return P
