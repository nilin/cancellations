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
from ..utilities.numutil import make_single_x
from ..utilities.tracking import dotdict
from ..plotting import plotting
from ..display import _display_
from . import plottools as pt
from . import exampleutil
import math
from functools import partial
from jax.lax import cond
from jax.tree_util import tree_map




#dontpick



####################################################################################################

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


####################################################################################################

class Lossgrad_memory:
    def __init__(self,momentumtimescale,g,rho,batchmode='momentum',**kw):
        g_=make_single_x(g)
        Dg_=jax.grad(g_)

        self.g=g
        self.Dg=jax.vmap(Dg_,in_axes=(None,0))
        self.rho=rho

        self.momentumtimescale=momentumtimescale

        match batchmode:
            case 'momentum':
                self.carryover=[(0,0,0)]
                eps=1/momentumtimescale
                a=1-eps
                self.update_E=partial(self.update_E_momentum,a,eps)

            case 'batch':
                self.carryover=[(0.0,0.0,0.0),(0.0,0.0,0.0),0]
                self.update_E=partial(self.update_E_batch,self.momentumtimescale)


    @classmethod
    def get_loss_and_grad(cls,g,Dg,rho,**kw):
        raise NotImplementedError

    def _eval_(self,params,X,fX):
        n=X.shape[0]
        h=n//2
        X1=X[:h]
        X2=X[h:]
        fX1=fX[:h]
        fX2=fX[h:]
        loss,G,*self.carryover=self.loss_and_grad(params,X1,X2,fX1,fX2,*self.carryover)
        return loss, G

    @staticmethod
    def update_E_momentum(a,eps,updates,Es):
        return [tuple(eps*u+a*E for u,E in zip(updates,Es))]

    @staticmethod
    def update_E_batch(ts,updates,Es,sums,i):
        check=(i%ts==0)*(i!=0)
        Es=tuple((i%ts!=0)*E+(i==0)*u+check*s/ts for u,E,s in zip(updates,Es,sums))
        sums=tuple((i%ts!=0)*(s+u) for u,s in zip(updates,sums))
        i+=1
        return Es, sums, i



class Lossgrad_balanced(Lossgrad_memory):

    def __init__(self,*a,mode='nonsquare',**kw):
        super().__init__(*a,**kw)
        self.loss_and_grad=jax.jit(partial(self.loss_and_grad,self.update_E,self.g,self.Dg,self.rho,mode))

    @staticmethod
    def loss_and_grad(update_E,g,Dg,rho,mode,params,X1,X2,f1,f2,Es,*_):

        g1,g2=g(params,X1),g(params,X2)
        Dg2=Dg(params,X2)
        rho1,rho2=rho(X1),rho(X2)

        #testing
        rho1,rho2=jnp.ones_like(rho1),jnp.ones_like(rho2)

        fg=jnp.average(f1*g1/rho1)
        ff=jnp.average(f1*f1/rho1)
        gg=jnp.average(g1*g1/rho1)

        assert((f1*g1/rho1).shape==rho1.shape)
        assert((f1*f1/rho1).shape==rho1.shape)
        assert((g1*g1/rho1).shape==rho1.shape)

        (Efg,Eff,Egg),*_=update_E((fg,ff,gg),Es,*_)

        match mode:
            case 'nonsquare':
                gradfactor=-1/jnp.exp(1.5*Egg)
            case 'square':
                gradfactor=-2*Efg/(Eff*Egg**2)
            case 'hopeforthebest':
                gradfactor=-jnp.ones((1,))

        gradweights=gg*f2/rho2*f2-fg*g2/rho2

        assert(gradfactor.size==1)
        assert(gradweights.shape==rho2.shape)

        grad=tree_map(lambda D:gradfactor*jnp.tensordot(gradweights,D,axes=(0,0)),Dg2)

        loss=1-fg*fg/(ff*gg)
        return loss,grad,(Efg,Eff,Egg),*_




class Lossgrad_separate_denominators(Lossgrad_memory):

    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.loss_and_grad=jax.jit(partial(self.loss_and_grad,self.update_E,self.g,self.Dg,self.rho))

    @staticmethod
    def loss_and_grad(update_E,g,Dg,rho,params,X1,X2,f1,f2,Es,*_):

        g1,g2=g(params,X1),g(params,X2)
        Dg2=Dg(params,X2)
        rho1,rho2=rho(X1),rho(X2)

        fg=jnp.average(f1*g1/rho1)
        ff=jnp.average(f1*f1/rho1)
        gg=jnp.average(g1*g1/rho1)

        gradweights1=-f2/rho2
        gradweights2=(fg/rho1)*g2/rho2

        assert(gradweights1.shape==rho2.shape)
        (Efg,Eff,Egg),*_=update_E((fg,ff,gg),Es,*_)

        Z=jnp.sqrt(Egg)
        Z3=Egg*Z

        grad=tree_map(lambda D:\
            jnp.tensordot(gradweights1,D,axes=(0,0))/Z\
            +jnp.tensordot(gradweights2,D,axes=(0,0))/Z3,\
            Dg2)

        loss=1-fg*fg/(ff*gg)
        return loss,grad,(Efg,Eff,Egg),*_

