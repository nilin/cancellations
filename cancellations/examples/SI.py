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

from cancellations.config import config as cfg, tracking
from cancellations.run import runtemplate, sampling
from cancellations.config.tracking import Profile, log




class Run(runtemplate.Run_statictarget):
    processname='Barron_norm'

    @staticmethod
    def getlearner(profile):
        #return getBarronfn(profile)
        return examples.getlearner_example(profile)

    @staticmethod
    def gettarget(P):
        #P.target=examples.getlearner_example(Profile(n=P.n,d=P.d,ndets=P.mtarget))
        P.target=examples.get_harmonic_oscillator2d(P)
        return P.target

    @classmethod
    def getdefaultprofile(cls,**kwargs):
        P=profile=super().getdefaultprofile(**kwargs)
        P.Y=P.Y/jnp.sqrt(jnp.average(P.Y**2/P.rho))
        samplespipe=sampling.SamplesPipe(profile.X,profile.Y,profile.rho,minibatchsize=profile.batchsize)
        profile.sampler=samplespipe.step
        checknorm=[]
        for i in range(10):
            X_,Y_,rho_=profile.sampler()
            checknorm.append(jnp.average(Y_**2/rho_))
        tracking.log('norm of target: {}'.format(jnp.average(jnp.array(checknorm))))
        Barron=P.learner
        log('generating lossgrad')
        profile.lossgrads=[\
            #get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_),.001),\
            #get_barronweight(1.0,Barron._eval_),\
            losses.get_lossgrad_NONSI(P.learner._eval_),\
            losses.get_lossgrad_SI(P.learner._eval_),\
            #losses.get_lossgrad_SI(P.learner._eval_)
        ]
        profile.lossnames=['non-SI','SI']
        #profile.lossweights=[]
        return profile

    @classmethod
    def getprofiles(cls):
        return {\
            'non-SI': partial(cls.getdefaultprofile,n=5,d=2,ndets=100,lossweights=[1.0,0.0]),
            'SI': partial(cls.getdefaultprofile,n=5,d=2,ndets=100,lossweights=[0.0,1.0]),
        }
