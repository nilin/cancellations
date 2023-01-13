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
from cancellations.examples import energy
from cancellations.examples import examples, losses

from cancellations.config.tracking import dotdict
from cancellations.run import unsupervised





class Run(unsupervised.Run):
    processname='energy'

    @classmethod
    def getdefaultprofile(cls):
        profile=super().getdefaultprofile().butwith(\
            getlearner=examples.get_harmonic_oscillator2d,\
            samples_train=10**5,\
            weight_decay=0.1)

        profile.initlossgrads=[losses.Lossgrad_SI,energy]
        profile.lossnames=['SI','E']
        profile.lossperiods=[1,1]

        return profile

    @classmethod
    def getprofiles(cls):
        default=cls.getdefaultprofile()
        profiles={'default':default}

        def getlearner2(profile):
            return Product(_functions_.IsoGaussian(1.0),_functions_.ComposedFunction(\
                _functions_.SingleparticleNN(**profile.learnerparams['SPNN']),\
                _functions_.Backflow(**profile.learnerparams['backflow']),\
                _functions_.Dets(n=profile.n,**profile.learnerparams['dets']),\
                _functions_.Sum()\
                ))
        p2=default.butwith(getlearner=getlearner2)
        p2.learnerparams=dotdict(\
            SPNN=dotdict(widths=[p2.d,25,25],activation='sp'),\
            backflow=dotdict(widths=[25,25,25],activation='sp'),\
            dets=dotdict(d=25,ndets=25),)
            #'OddNN':dict(widths=[25,1],activation='sp')
        profiles['backflow']=p2

        return profiles
