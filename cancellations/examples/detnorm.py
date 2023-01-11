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

from cancellations.config.tracking import dotdict
from cancellations.run import runtemplate


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################





class Run(runtemplate.Run):
    processname='det_norm'

    @classmethod
    def getdefaultprofile(cls):
        profile=super().getdefaultprofile()
        profile.target=examples.get_harmonic_oscillator2d(profile)
        profile.learner=runtemplate.getlearner_example(profile)
        profile.lossgrads=[losses.get_lossgrad_SI(profile.learner._eval_,profile)]
        profile.lossnames=['SI']
        profile.lossweights=[1.0]
        profile.sampler=profile.getXYsampler(profile.Xsampler,profile.target)
        return profile
