#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses
import jax.numpy as jnp
from functools import partial
import jax
import os
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_leaves
import matplotlib.pyplot as plt

from cancellations.functions import _functions_, examplefunctions as examples
from cancellations.functions._functions_ import Product
from cancellations.tracking.browse import Browse
from cancellations.tracking.tracking import dotdict
from cancellations.utilities.textutil import printtree

from cancellations.tracking import runconfig as cfg, tracking, sysutil
from cancellations.run import runtemplate, sampling
from cancellations.tracking.tracking import dotdict, log




class Run(runtemplate.Fixed_XY):
    
    @classmethod
    def getprofile(cls,parentprocess):
        #mode=parentprocess.browse(options=['non-SI','SI loss','SI_2','SI_3','SI_4'])

        n,d,samples_train,minibatchsize=3,2,10**5,2
        target=examples.get_harmonic_oscillator2d(n,d)
        learner=cls.getlearner_example(n,d)

        balanced=parentprocess.browse(options=[True,False])

        #symmode=parentprocess.browse(options=['antisym','nonsym'])
        #if symmode=='nonsym':
        #    target,_=_functions_.switchtype(target)
        #    learner,_=_functions_.switchtype(learner)

        P=super().getprofile(n,d,samples_train,minibatchsize,target).butwith(learner=learner,balanced=balanced)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        g=P.learner._eval_

        log(losses.norm(P.Y,P.rho))
        cls.normalizelearner(P)
        log(losses.norm(P.learner.eval(P.X),P.rho))

        #match mode:
        #    case 'L2':
        #        lossgrad=losses.get_lossgrad_NONSI(g)
        #    case 'SI loss':
        #        lossgrad=losses.get_lossgrad_SI(g)
        #    case 'SI_2':
        #        lossgrad=SI(g,timescale=10,normpow=2).lossgrad
        #    case 'SI_3':
        #        lossgrad=SI(g,timescale=10,normpow=3).lossgrad
        #    case 'SI_4':
        #        lossgrad=SI(g,timescale=10,normpow=0,olpow=0).lossgrad

        T=lambda x:jnp.log(x)**2
        #T=lambda x:jnp.exp(jax.nn.relu(jnp.abs(jnp.log(x))-7.0)),\
        #T=lambda x:abs(x)**(1/7),\


        P.lossweights=[1.0]
        P.lossgrads=[\
            #losses.transform_grad(lambda params,X,Y,rho: losses.norm(g(params,X),rho),T=T,fromrawloss=True),\
            SGD(g,P.balanced,10).norm_gradL,\
            losses.get_lossgrad_SI(g),\
            losses.get_lossgrad_NONSI(g),\
            Hloss(g).lossgrad\
        ]
        P.lossnames=['norm','SI loss','L2','second derivative estimate']

        P.info['learner']=P.learner.getinfo()
        P.info['balanced']=P.balanced
        #P.name=mode
        return P

    @staticmethod
    def getlearner_example(n,d):
        lprofile=tracking.dotdict(\
            SPNN=dotdict(widths=[d,10,50],activation='sp'),\
            #backflow=dotdict(widths=[50,50],activation='sp'),\
            dets=dotdict(d=50,ndets=100),)

        return _functions_.Product(_functions_.ScaleFactor(),_functions_.IsoGaussian(1.0),_functions_.ComposedFunction(\
            _functions_.SingleparticleNN(**lprofile['SPNN']),\
            #_functions_.Backflow(**lprofile['backflow']),\
            _functions_.Dets(n=n,**lprofile['dets']),\
            _functions_.Sum()\
            ))

    @staticmethod
    def normalizelearner(P,N=1000):
        log('normalizing learner')
        params=P.learner.weights
        params[0]/=losses.norm(P.learner.eval(P.X[:N]),P.rho[:N])
        
class Plot(Run):
    def execprocess(self):
        super().plot(None,currentrun=False)
    @staticmethod
    def getprofile(*a,**kw): return tracking.dotdict()
    


def value_and_grad_mapped(f):
    f_single_x=lambda params,x:jnp.squeeze(f(params,jnp.expand_dims(x,axis=0)))
    return jax.vmap(jax.value_and_grad(f_single_x),in_axes=(None,0))





class SGD:
    def __init__(self,f,balanced,timescale):
        self.sqZ=1.0
        self.e=1/timescale
        self.a=1-self.e
        self.G=partial(self.sq_and_gradL,f,balanced=balanced)
        
    @staticmethod
    def sq_and_gradL(f,params,X,Y,rho,Z,balanced):
        X1,g1,rho1=X[0:1],Y[0],rho[0]
        X2,g2,rho2=X[1:2],Y[1],rho[1]

        f1=jnp.squeeze(f(params,X1))
        f2,Df2=jax.value_and_grad(lambda params_,X_: jnp.squeeze(f(params_,X_)))(params,X2)

        if balanced:
            G=(f1**2*g2-f1*g1*f2)/Z**3
        else:
            G=(g2/Z-f1*g1*f2/Z**3)

        return (f1**2+f2**2)/2,tree_map(lambda A:G*A,Df2)
        
    def norm_gradL(self,params,X,Y,rho):
        Z=jnp.sqrt(self.sqZ)
        sq,G=self.G(params,X,Y,rho,Z)
        self.sqZ=self.a*self.sqZ+self.e*sq
        return Z,G

#
#

class Hloss:
    def __init__(self,f):
        self.SI_loss=losses.get_lossgrad_SI(f)
        self.prevG=None
        self.prevparams=None

    def lossgrad(self,params,X,Y,rho):
        _,G=self.SI_loss(params,X,Y,rho)
        try:
            dG=self.squared_difference(G,self.prevG)
            dtheta=self.squared_difference(params,self.prevparams)
            loss=jnp.sqrt(dG/dtheta)
        except:
            loss=0
        self.prevG=G
        self.prevparams=params
        return loss, self.emptyparams(params)

    @staticmethod
    def squared_difference(A,B):
        squared_difference_tree=tree_map(lambda a,b: jnp.sum((a-b)**2),A,B)
        return tree_reduce(lambda a,b: a+b, squared_difference_tree)

    @staticmethod
    def emptyparams(params):
        return tree_map(lambda g:jnp.zeros_like(g),params)

#
#class SI:
#    def __init__(self,f,timescale=100,normpow=3,olpow=0):
#        self.grad_etc=jax.jit(lambda params,X,Y,rho: self.norm_and_overlap_and_grad(f,params,X,Y,1/rho))
#        self.eps=1/timescale
#        self.r=1-self.eps
#        self.normpow=normpow
#        self.olpow=olpow
#        self.ff=None
#        self.yf=None
#
#    def lossgrad(self,params,X,Y,rho):
#        normsq,overlap,G=self.grad_etc(params,X,Y,rho)
#        self.ff=normsq if self.ff is None else normsq**self.eps*self.ff**self.r
#        self.yf=overlap if self.yf is None else overlap**self.eps+self.r*self.yf**self.r
#        #return 1-overlap**2/self.Z**2,tree_map(lambda g:g/self.Z**self.npow,G)
#        return jnp.sqrt(self.ff),tree_map(lambda g:g*self.yf**self.olpow/self.ff**(self.normpow/2),G)
#
#    @staticmethod
#    def norm_and_overlap_and_grad(f,params,X,Y,weights):
#        fX,GfX=value_and_grad_mapped(f)(params,X)
#        normsq=jnp.average(fX**2*weights)
#        overlap=jnp.average(fX*Y*weights)
#        a=-normsq*Y+overlap*fX
#        G=tree_map(lambda g:jnp.tensordot(a*weights,g,axes=(0,0)),GfX)
#        return normsq,overlap,G