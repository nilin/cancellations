#
# nilin
# 
# 2022/7
#


from re import I
import jax
import jax.numpy as jnp
import jax.random as rnd
from ..functions import examplefunctions as ef, examplefunctions3d, functions
from ..learning import learning
from ..functions.functions import ComposedFunction,SingleparticleNN,Product
from ..utilities import config as cfg, numutil, tracking, sysutil, textutil, sampling, setup
from ..utilities.tracking import dotdict
from ..plotting import plotting
from ..display import _display_
from . import plottools as pt
from . import exampleutil
import math
from functools import partial
from . import losses

from . import exampletemplate





class Run(exampletemplate.Run):
    processname='harmonicoscillator2d'

    @classmethod
    def getdefaultprofile(cls):
        return super().getdefaultprofile().butwith(gettarget=gettarget)

    @classmethod
    def getprofiles(cls):
        profiles=dict()
        default=cls.getdefaultprofile().butwith(n=6,weight_decay=.1)
        profiles['n=6 d=2 SI']=default
        profiles['n=6 d=2 non-SI']=default.butwith(initlossgrad=losses.Lossgrad_nonSI)
        profiles['n=6 d=2 unbiased SI loss']=\
            {\
                'momentum 10':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_unbiased,10),\
                    minibatchsize=200),\
                'momentum 1':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_unbiased,1),\
                    minibatchsize=200),\
                'small mb, momentum 10':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_unbiased,10),\
                    minibatchsize=10),\
                'small mb, momentum 1':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_unbiased,1),\
                    minibatchsize=10),\
                'small mb, reference (biased)':default.butwith(\
                    minibatchsize=5),\
            }
        return profiles

def gettarget(P,run):
    f=functions.Slater(*['psi{}_{}d'.format(i,P.d) for i in range(1,P.n+1)])
    f=normalize(f, run.genX, P.X_density)
    ftest=normalize(f, run.genX, P.X_density)
    run.log('double normalization factor check (should~1) {:.3f}'.format(ftest.elements[0].weights))
    return f

def normalize(f,genX,Xdensity):
    C=functions.ScaleFactor()

    X=genX(1000)
    rho=Xdensity(X)
    Y=f.eval(X)
    assert(Y.shape==rho.shape)
    squarednorm=jnp.average(Y**2/rho)
    C.weights=1/jnp.sqrt(squarednorm)

    return Product(C,f)
