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


class Lossgrad_unbiased:
    def __init__(self,momentumtimescale,g,rho):
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

            gradfactor=-2*Efg/(Eff*Egg**2)

            fg=jnp.average(rho1*f1*g1)
            fs=jnp.average(rho1*f1*f1)
            gs=jnp.average(rho1*g1*g1)

            gradweights=gs*rho2*f2-fg*rho2*g2

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


#    def _eval_(self,params,X,fX):
#        x1=X[0]
#        x2=X[1]
#        f1=fX[0]
#        f2=fX[1]
#        loss,G,f_,g_=self.loss_and_grad_single_x(params,x1,x2,f1,f2,*self.carryover)
#        self.update_E(f_,g_)
#        return loss, G
#

#class Lossgrad_unbiased:
#    def __init__(self,momentumtimescale,learnerfn,rho):
#        self.learnerfn=learnerfn
#
#        g=make_single_x(self.learnerfn)
#        Dg=jax.grad(g)
#
#        self.eps=1/momentumtimescale
#        self.a=1-self.eps
#
#        self.i=0
#        self.carryover=[0,1,1]
#
#        def loss_and_grad_single_x(params,x1,x2,f1,f2,Efg,Eff,Egg):
#
#            g1,g2=g(params,x1),g(params,x2)
#            Dg2=Dg(params,x2)
#            rho1,rho2=rho(x1),rho(x2)
#
#            gradfactor=rho1*rho2*(g1**2*f2-f1*g1*g2)
#            gradfactor*=-2*Efg/(Eff*Egg**2)
#
#            loss=1-Efg**2/(Eff*Egg)
#
#            if setup.debug:
#                breakpoint()
#            grad=tree_map(lambda A:gradfactor*A,Dg2)
#
#    def update_E(self,rho1,rho2,f1,f2,g1,g2):
#        Efg,Eff,Egg=self.carryover
#
#        a,eps=0,1 if self.i==0 else self.a, self.eps
#        Efg=a*Efg+eps*(rho1*f1/2+rho2*f2/2)*(rho1*g1/2+rho2*g2/2)
#        Eff=a*Eff+eps*(rho1*f1/2+rho2*f2/2)**2
#        Egg=a*Egg+eps*(rho1*g1/2+rho2*g2/2)**2
#
#        self.carryover=(Efg,Eff,Egg)
#        self.i+=1
#    def update_E(self,rho1,rho2,f1,f2,g1,g2):
#        Efg,Eff,Egg=self.carryover
#
#        a,eps=0,1 if self.i==0 else self.a, self.eps
#        Efg=a*Efg+eps*(rho1*f1/2+rho2*f2/2)*(rho1*g1/2+rho2*g2/2)
#        Eff=a*Eff+eps*(rho1*f1/2+rho2*f2/2)**2
#        Egg=a*Egg+eps*(rho1*g1/2+rho2*g2/2)**2
#
#        self.carryover=(Efg,Eff,Egg)
#        self.i+=1
#
#    def _eval_(self,params,X,fX):
#        n=X.shape[0]
#        h=n//2
#        X1=X[:h]
#        X2=X[h:]
#        fX1=fX[:h]
#        fX2=fX[h:]
#        loss,G,*self.carryover=self.loss_and_grad(params,X1,X2,fX1,fX2,*self.carryover,self.i)
#        self.update_E()
#        return loss, G
#