#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses
import jax.numpy as jnp
from functools import partial
import jax
from jax.tree_util import tree_map

from cancellations.functions import _functions_, examplefunctions as examples
from cancellations.functions._functions_ import Product

import math
from cancellations.config import config as cfg
from cancellations.run import runtemplate
from cancellations.config.tracking import Profile, log, sysutil
import matplotlib.pyplot as plt
import os


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################


def getBarronfn(P):
    match P.mode:
        case 'ANTISYM':
            return Product(_functions_.IsoGaussian(1.0),\
                _functions_.ASBarron(n=P.n,d=P.d,m=P.m,ac=P.ac))
        case 'RAW':
            return Product(_functions_.IsoGaussian(1.0),\
                _functions_.Barron(n=P.n,d=P.d,m=P.m,ac=P.ac))


class Run(runtemplate.Run_statictarget):
    processname='Barron_norm'

    @staticmethod
    def getlearner(profile):
        return getBarronfn(profile)

    @staticmethod
    def gettarget(P):
        #P.target=examples.getlearner_example(Profile(n=P.n,d=P.d,ndets=P.mtarget))
        P.target=examples.get_harmonic_oscillator2d(P)

    @classmethod
    def getdefaultprofile(cls,ac='softplus',**kwargs):
        P=profile=super().getdefaultprofile(ac=ac,**kwargs)
        Barron=P.learner
        log('generating lossgrad')
        profile.lossgrads=[\
            #get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_,lambda x:jnp.tan((math.pi/2)*x)),.00001),\
            get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_),.00001),\
            get_barronweight(1.0,Barron._eval_),\
        ]
        profile.lossnames=['eps','Barron norm estimate']
        profile.lossweights=[100.0,0.01]
        return profile

    @classmethod
    def getprofiles(cls):
        return {\
            #'m=1': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=1),
            #'m=2': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=2),
            #'m=4': partial(cls.getdefaultprofile,n=5,d=2,m=100,mtarget=4),
            'ANTISYM n=5': partial(cls.getdefaultprofile,n=5,d=2,m=200,mode='ANTISYM',batchsize=25),\
            'ANTISYM relu n=5': partial(cls.getdefaultprofile,n=5,d=2,m=200,mode='ANTISYM',ac='relu',batchsize=25),\
            'RAW n=5': partial(cls.getdefaultprofile,n=5,d=2,m=10*math.factorial(5),mode='RAW'),\
            'RAW relu n=5': partial(cls.getdefaultprofile,n=5,d=2,m=10*math.factorial(5),mode='RAW',ac='relu'),\
            'ANTISYM n=4': partial(cls.getdefaultprofile,n=4,d=2,m=500,mode='ANTISYM'),\
            'ANTISYM relu n=4': partial(cls.getdefaultprofile,n=4,d=2,m=1000,mode='ANTISYM',ac='relu'),\
            'RAW relu n=4': partial(cls.getdefaultprofile,n=4,d=2,m=1000*math.factorial(5),mode='RAW',ac='relu'),\
            'RAW relu n=4 small': partial(cls.getdefaultprofile,n=4,d=2,m=10*math.factorial(5),mode='RAW',ac='relu'),\
            'RAW n=4 small': partial(cls.getdefaultprofile,n=4,d=2,m=10*math.factorial(5),mode='RAW'),\
            'RAW relu n=4 medium': partial(cls.getdefaultprofile,n=4,d=2,m=100*math.factorial(5),mode='RAW',ac='relu'),\
            'ANTISYM n=3': partial(cls.getdefaultprofile,n=3,d=2,m=1000,mode='ANTISYM'),\
            'RAW n=3': partial(cls.getdefaultprofile,n=3,d=2,m=1000*math.factorial(3),mode='RAW'),\
            'ANTISYM relu n=3': partial(cls.getdefaultprofile,n=3,d=2,m=1000,mode='ANTISYM',ac='relu'),\
            'RAW relu n=3': partial(cls.getdefaultprofile,n=3,d=2,m=1000*math.factorial(3),mode='RAW',ac='relu'),\
        }

    def plot(self,P):
        #fig,(ax0,ax1)=plt.subplots(2,1,figsize=(7,15))
        fig,(ax0,ax1)=plt.subplots(2,1)
        plt.rcParams['text.usetex']
        fig.suptitle('{} Barron norm, n={}, m={}, {}'.format(P.mode,P.n,P.m,P.ac))
        Bnorms=jnp.array(self.losses['Barron norm estimate'])
        Bnorm=jnp.quantile(Bnorms[-1000:],.5)
        ax0.plot(Bnorms,'b',label='Barron Ansatz weight norm')
        #ax0.plot(self.losses['eps'],'r:',label='$\epsilon$')
        #ax0.set_yscale('log')
        ax0.set_ylim(0,Bnorm*3)
        ax0.legend()
        epss=jnp.array(self.losses['eps'])
        eps=jnp.quantile(epss[-1000:],.5)
        ax1.plot(epss,'r',label='$\epsilon$')
        ax1.plot(eps*jnp.ones_like(epss),'k:')
        if eps<.1:
            ax0.plot(Bnorm*jnp.ones_like(Bnorms),'k:',label='$\epsilon$-smooth Barron norm estimate')
        ax1.set_yscale('log')
        ax1.legend()
        outpath=os.path.join('plots','Bnorm_{}_n={}_{}___{}.pdf'.format(P.mode,P.n,P.ac,cfg.session.ID))
        sysutil.savefig(outpath)
        sysutil.showfile(outpath)

def get_barronweight(p,f):
    def norm(params,X,rho):
        return jnp.sqrt(jnp.average(f(params,X)**2/rho))
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
    lossfn=lambda params,X,Y,rho: loss(p,params)/norm(params,X,rho)
    return jax.jit(jax.value_and_grad(lossfn))

def get_threshold_lg(lg,delta):
    def _eval_(params,*X):
        val,grad=lg(params,*X)
        weight=jax.nn.sigmoid(val/delta-1)
        return val,tree_map(lambda A:weight*A,grad)
    return jax.jit(_eval_)
