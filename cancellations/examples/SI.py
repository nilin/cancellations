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
from cancellations.config.browse import Browse

from cancellations.config import config as cfg, tracking
from cancellations.run import runtemplate, sampling
from cancellations.config.tracking import Profile, log




class Run(runtemplate.Fixed_XY):
    processname='Barron_norm'
    
    def getprofile(self):
        mode=tracking.runprocess(Browse(options=['non-SI','SI']))

        n,d,samples_train,minibatchsize=5,3,10**5,25
        target=examples.get_harmonic_oscillator2d(n,d)
        P=super().getprofile(n,d,samples_train,minibatchsize,target)
        P.Y=P.Y/jnp.sqrt(jnp.average(P.Y**2/P.rho))
        self.initsampler(P)
        checknorm=[]
        for i in range(10):
            X_,Y_,rho_=P.sampler.sample()
            checknorm.append(jnp.average(Y_**2/rho_))
        log('norm of target: {}'.format(jnp.average(jnp.array(checknorm))))
        log('generating learner and lossgrad')
        P.learner=examples.getlearner_example(n,d)
        P.lossgrads=[\
            #get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_),.001),\
            #get_barronweight(1.0,Barron._eval_),\
            losses.get_lossgrad_NONSI(P.learner._eval_),\
            losses.get_lossgrad_SI(P.learner._eval_),\
            #losses.get_lossgrad_SI(P.learner._eval_)
        ]
        P.lossnames=['non-SI','SI']
        match mode:
            case 'non-SI':
                P.lossweights=[1.0,0.0]
            case 'SI':
                P.lossweights=[0.0,1.0]
        return P

