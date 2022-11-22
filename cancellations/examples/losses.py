#
# nilin
# 
# 2022/7
#


import jax
import jax.numpy as jnp
import jax.random as rnd
from ..utilities import numutil, tracking
from ..utilities.numutil import make_single_x
from functools import partial
from jax.tree_util import tree_map
from cancellations.functions import functions



#dontpick



####################################################################################################

class Lossgrad:
    pass

class Lossgrad_SI(Lossgrad):
    def __init__(self,fd,density):
        f=fd._eval_
        self.lossfn=lambda params,X,Y: numutil.weighted_SI_loss(f(params,X),Y,relweights=1/density(X))
        self._eval_=jax.jit(jax.value_and_grad(self.lossfn))

class Lossgrad_nonSI(Lossgrad):
    def __init__(self,fd,density):
        f=fd._eval_
        def lossfn(params,X,Y):
            rho=density(X)
            sqdist=(f(params,X)-Y)**2
            assert(sqdist.shape==rho.shape)
            return jnp.average(sqdist/rho)
        self.lossfn=lossfn
        self._eval_=jax.jit(jax.value_and_grad(self.lossfn))


#class Lossgrad_sum(Lossgrad):
#    def __init__(self,*lossgrads)
#        self.lossgrads=lossgrads
#
#    def _eval_(self,params,X,fX):
#        losses,grads=zip(*[l._eval_(params,X,fX) for l in self.lossgrads])
#        grad=grads[0]
#        for G in grads[1:]:
#            grad=tree_map(lambda D1+D2:,grads,)
#        return jnp.sum(losses),grad

####################################################################################################

class Lossgrad_memory(Lossgrad):
    def __init__(self,period,microbatchsize,gd,rho,batchmode='batch',**kw):
        g=gd._eval_
        g_=make_single_x(g)
        Dg_=jax.grad(g_)

        self.g=g
        self.Dg=jax.vmap(Dg_,in_axes=(None,0))
        self.rho=rho

        self.period=period
        self.microbatchsize=microbatchsize

        match batchmode:
            case 'batch':
                self.update_E=partial(self.update_E_batch,g,rho)
                self.i=0

    def _eval_(self,params,X,fX):
        n=X.shape[0]
        h=self.microbatchsize
        assert(2*h<=n)

        I=rnd.choice(tracking.nextkey(),n,(2*h,))
        X=X[I]
        fX=fX[I]

        if self.i%self.period==0:
            self.Es=self.update_E(params,X,fX)
            #tracking.log('update Es')

        #tracking.log(type(X))
        #tracking.log(type(fX))

        X1=X[:h]
        X2=X[h:2*h]
        fX1=fX[:h]
        fX2=fX[h:2*h]

        Efg,Eff,Egg=self.Es
        G=self.gradestimate(params,X1,X2,fX1,fX2,*self.Es)
        loss=1-Efg*Efg/(Eff*Egg)

        self.i+=1
        return loss, G

    @staticmethod
    def update_E_batch(g,rho,params,X,fX):
        gX=g(params,X)
        rhoX=rho(X)
        Efg=jnp.average(fX*gX/rhoX)
        Eff=jnp.average(fX*fX/rhoX)
        Egg=jnp.average(gX*gX/rhoX)

        return (Efg,Eff,Egg)



class Lossgrad_balanced(Lossgrad_memory):

    def __init__(self,period,microbatchsize,gd,rho,mode='nonsquare',**kw):
        super().__init__(period,microbatchsize,gd,rho,**kw)
        self.gradestimate=jax.jit(partial(self.gradestimate,self.g,self.Dg,self.rho,mode))

    @staticmethod
    def gradestimate(g,Dg,rho,mode,params,X1,X2,f1,f2,Eff,Efg,Egg):

        g1,g2=g(params,X1),g(params,X2)
        Dg2=Dg(params,X2)
        rho1,rho2=rho(X1),rho(X2)

        #testing
        rho1,rho2=jnp.ones_like(rho1),jnp.ones_like(rho2)

        fg=jnp.average(f1*g1/rho1)
        gg=jnp.average(g1*g1/rho1)

        assert((f1*g1/rho1).shape==rho1.shape)
        assert((f1*f1/rho1).shape==rho1.shape)
        assert((g1*g1/rho1).shape==rho1.shape)

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

        return tree_map(lambda D:gradfactor*jnp.tensordot(gradweights,D,axes=(0,0)),Dg2)


####################################################################################################


class Lossgrad_normratio(Lossgrad):
    def __init__(self,learnerdescr,density,fpow=2,Afpow=-2):

        Af=learnerdescr._eval_
        fdescr,switchcounts=functions.switchtype(learnerdescr)
        assert(switchcounts==1)
        f=fdescr._eval_

        fnorm=partial(self.getnorm,f,density=density)
        Afnorm=partial(self.getnorm,Af,density=density)

        self.lossfn=lambda params,X,_: jnp.exp((fpow/2)*jnp.log(fnorm(params,X)))*jnp.exp((Afpow/2)*jnp.log(Afnorm(params,X)))
        self._eval_=jax.jit(jax.value_and_grad(self.lossfn))


    @staticmethod
    def getnorm(_f_,weights,X,density):
        Y=_f_(weights,X)
        Xdensity=density(X)
        squaresums=jnp.sum(Y**2,axis=Y.shape[1:])
        assert(squaresums.shape==Xdensity.shape)
        normalized=squaresums/Xdensity
        return jnp.average(normalized)
