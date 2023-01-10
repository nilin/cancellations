#
# nilin
# 
# 2022/7
#


import jax.numpy as jnp
from functools import partial

from cancellations.functions import _functions_
from cancellations.functions._functions_ import Product
from cancellations.lossesandnorms import losses,losses2
from cancellations.examples import runtemplate, harmonicoscillator2d


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################


def getlearner(profile):
    return Product(_functions_.IsoGaussian(1.0),\
        _functions_.ASBarron(n=profile.n,d=profile.d,m=100))
        #_functions_.ASBarron(**profile.learnerparams['ASNN']))


class Run(runtemplate.Run):
    processname='Barron_norm'

    @classmethod
    def getdefaultprofile(cls):
        return super().getdefaultprofile().butwith(\
            getlearner=getlearner,\
            gettarget=harmonicoscillator2d.gettarget,\
            samples_train=10**5)
