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
        default=cls.getdefaultprofile().butwith(n=6,weight_decay=.1,iterations=10**4)
        ts=1
        profiles['n=6 d=2 balanced SI loss (batch)']=\
            {\
                'balanced, small mb, batch {}'.format(ts):default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_balanced,ts,mode='nonsquare',batchmode='batch'),\
                    minibatchsize=10),\
                'balanced, squared, small mb, batch 100':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_balanced,100,mode='square',batchmode='batch'),\
                    minibatchsize=10),\
                'balanced, hopeforthebest, small mb, batch 100':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_balanced,100,mode='hopeforthebest',batchmode='batch'),\
                    minibatchsize=10),\
                'separate denominators, small mb, batch 100':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_separate_denominators,100,batchmode='batch'),\
                    minibatchsize=10),\
                'small mb, reference (biased)':default.butwith(\
                    minibatchsize=5),\
            }
        profiles['n=6 d=2 balanced SI loss (momentum)']=\
            {\
                'balanced, small mb, momentum {}'.format(ts):default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_balanced,ts,mode='nonsquare'),\
                    minibatchsize=10),\
                'balanced, squared, small mb, momentum 100':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_balanced,100,mode='square'),\
                    minibatchsize=10),\
                'balanced, hopeforthebest, small mb':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_balanced,100,mode='hopeforthebest'),\
                    minibatchsize=10),\
                'separate denominators, small mb, momentum 100':default.butwith(\
                    initlossgrad=partial(losses.Lossgrad_separate_denominators,100),\
                    minibatchsize=10),\
                'small mb, reference (biased)':default.butwith(\
                    minibatchsize=5),\
            }
        profiles['n=6 d=2 SI']=default
        profiles['n=6 d=2 non-SI']=default.butwith(initlossgrad=losses.Lossgrad_nonSI)
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
