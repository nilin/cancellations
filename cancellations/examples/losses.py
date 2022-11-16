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




#dontpick

class Lossgrad:
    pass

class Lossgrad_SI(Lossgrad):
    def __init__(self,f,density):
        self.lossfn=lambda params,X,Y: numutil.weighted_SI_loss(f(params,X),Y,relweights=1/density(X))
        self._eval_=jax.jit(jax.value_and_grad(self.lossfn))

class Lossgrad_nonSI(Lossgrad):
    def __init__(self,f,density):
        def lossfn(params,X,Y):
            rho=density(X)
            sqdist=(f(params,X)-Y)**2
            assert(sqdist.shape==rho.shape)
            return jnp.average(sqdist/rho)
        self.lossfn=lossfn
        self._eval_=jax.jit(jax.value_and_grad(self.lossfn))



def make_single_x(F):
    return lambda *x: jnp.squeeze(F(*x[:-1],jnp.expand_dims(x[-1],axis=0)))

from jax.tree_util import tree_map


class Lossgrad_balanced:
    def __init__(self,momentumtimescale,g,rho,mode='nonsquare'):
        g_=make_single_x(g)
        Dg_=jax.grad(g_)
        Dg=jax.vmap(Dg_,in_axes=(None,0))

        self.eps=1/momentumtimescale
        self.a=1-self.eps

        self.i=0
        self.carryover=[0,1,1]

        def loss_and_grad(params,X1,X2,f1,f2,Efg,Eff,Egg):

            g1,g2=g(params,X1),g(params,X2)
            Dg2=Dg(params,X2)
            rho1,rho2=rho(X1),rho(X2)

            match mode:
                case 'nonsquare':
                    gradfactor=-1/jnp.exp(1.5*Egg)
                case 'square':
                    gradfactor=-2*Efg/(Eff*Egg**2)
                case 'hopeforthebest':
                    gradfactor=-jnp.ones((1,))


            fg=jnp.average(f1*g1/rho1)
            fs=jnp.average(f1*f1/rho1)
            gs=jnp.average(g1*g1/rho1)

            gradweights=(gs/rho2)*f2-(fg/rho2)*g2

            assert(gradfactor.size==1)
            assert(gradweights.shape==rho2.shape)

            grad=tree_map(lambda D:gradfactor*jnp.tensordot(gradweights,D,axes=(0,0)),Dg2)

            loss=1-Efg**2/(Eff*Egg)
            return loss,grad,fg,fs,gs

        self.loss_and_grad=jax.jit(loss_and_grad)


    def _eval_(self,params,X,fX):
        n=X.shape[0]
        h=n//2
        X1=X[:h]
        X2=X[h:]
        fX1=fX[:h]
        fX2=fX[h:]
        loss,G,fg,fs,gs=self.loss_and_grad(params,X1,X2,fX1,fX2,*self.carryover)
        self.update_E(fg,fs,gs)
        return loss, G


    def update_E(self,fg,fs,gs):
        Efg,Eff,Egg=self.carryover

        a,eps=(0,1) if self.i==0 else (self.a, self.eps)
        Efg=a*Efg+eps*fg
        Eff=a*Eff+eps*fs
        Egg=a*Egg+eps*gs

        self.carryover=(Efg,Eff,Egg)
        self.i+=1

