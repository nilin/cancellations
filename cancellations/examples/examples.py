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
from cancellations.config import tracking
from cancellations.config.tracking import dotdict



def get_harmonic_oscillator2d(P):
    return _functions_.Slater(*['psi{}_{}d'.format(i,P.d) for i in range(1,P.n+1)])


def getlearner_example(profile):
    P=profile
    profile.learnerparams=tracking.dotdict(\
        SPNN=dotdict(widths=[profile.d,100,100],activation='sp'),\
        backflow=dotdict(widths=[100,100,100],activation='sp'),\
        dets=dotdict(d=100,ndets=P.ndets),)
        #'OddNN':dict(widths=[25,1],activation='sp')

    return Product(_functions_.IsoGaussian(1.0),_functions_.ComposedFunction(\
        _functions_.SingleparticleNN(**profile.learnerparams['SPNN']),\
        _functions_.Backflow(**profile.learnerparams['backflow']),\
        _functions_.Dets(n=profile.n,**profile.learnerparams['dets']),\
        _functions_.Sum()\
        ))