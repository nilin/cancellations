#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses
import jax.numpy as jnp
from functools import partial
import jax
import os
from jax.tree_util import tree_map
import matplotlib.pyplot as plt

from cancellations.functions import _functions_, examplefunctions as examples
from cancellations.functions._functions_ import Product
from cancellations.config.browse import Browse

from cancellations.config import config as cfg, tracking, sysutil
from cancellations.run import runtemplate, sampling
from cancellations.config.tracking import Profile, log




class Run(runtemplate.Fixed_XY):
    lossnames=['non-SI','SI','norm']
    
    def getprofile(self):
        mode=tracking.runprocess(Browse(options=['non-SI','SI']))

        n,d,samples_train,minibatchsize=5,3,10**5,25
        target=examples.get_harmonic_oscillator2d(n,d)
        P=super().getprofile(n,d,samples_train,minibatchsize,target)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        self.initsampler(P)

        log('generating learner and lossgrad')
        P.learner=self.getlearner_example(n,d)
        g=P.learner._eval_

        log(losses.norm(P.Y,P.rho))
        self.normalizelearner(P)
        log(losses.norm(P.learner.eval(P.X),P.rho))

        P.lossgrads=[\
            #get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_),.001),\
            #get_barronweight(1.0,Barron._eval_),\
            losses.get_lossgrad_NONSI(g),\
            losses.get_lossgrad_SI(g),\
            losses.transform_grad(\
                lambda params,X,Y,rho: losses.norm(g(params,X),rho),\
                T=lambda x:jnp.abs(jnp.log(x)),\
                fromrawloss=True)
            #losses.get_lossgrad_SI(P.learner._eval_)
        ]
        P.lossnames=self.lossnames #['non-SI','SI','norm']
        match mode:
            case 'non-SI':
                P.lossweights=[1.0,0.0,1.0]
            case 'SI':
                P.lossweights=[0.0,1.0,1.0]
        P.info['learner']=P.learner.getinfo()
        P.name=mode
        return P

    @staticmethod
    def getlearner_example(n,d):
        lprofile=examples.getlearner_example_profile(n,d)
        return _functions_.Product(_functions_.ScaleFactor(),_functions_.IsoGaussian(1.0),_functions_.ComposedFunction(\
            _functions_.SingleparticleNN(**lprofile['SPNN']),\
            _functions_.Backflow(**lprofile['backflow']),\
            _functions_.Dets(n=n,**lprofile['dets']),\
            _functions_.Sum()\
            ))

    @staticmethod
    def normalizelearner(P,N=1000):
        log('normalizing learner')
        params=P.learner.weights
        params[0]/=losses.norm(P.learner.eval(P.X[:N]),P.rho[:N])
        
class Plot(Run):
    def execprocess(self):
        super().plot(None,False)
    @staticmethod
    def getprofile(*a,**kw): return tracking.Profile()
    