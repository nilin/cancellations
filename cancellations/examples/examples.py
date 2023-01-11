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
from cancellations.run import supervised



def get_harmonic_oscillator2d(P):
    return _functions_.Slater(*['psi{}_{}d'.format(i,P.d) for i in range(1,P.n+1)])



