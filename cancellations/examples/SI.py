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
from cancellations.config.browse import Browse
from cancellations.config.tracking import dotdict
from cancellations.utilities.textutil import printtree

from cancellations.config import config as cfg, tracking, sysutil
from cancellations.run import runtemplate, sampling
from cancellations.config.tracking import Profile, log




class Run(runtemplate.Fixed_XY):
    lossnames=['SI','norm','custom','second_derivative']
    
    @classmethod
    def getprofile(cls,parentprocess):
        mode=parentprocess.browse(options=['non-SI','SI','SI_2','SI_3','SI_4'])

        n,d,samples_train,minibatchsize=4,2,10**5,100
        target=examples.get_harmonic_oscillator2d(n,d)
        P=super().getprofile(n,d,samples_train,minibatchsize,target)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        log('generating learner and lossgrad')
        P.learner=cls.getlearner_example(n,d)
        g=P.learner._eval_

        log(losses.norm(P.Y,P.rho))
        cls.normalizelearner(P)
        log(losses.norm(P.learner.eval(P.X),P.rho))

        match mode:
            case 'non-SI':
                lossgrad=losses.get_lossgrad_NONSI(g)
            case 'SI':
                lossgrad=losses.get_lossgrad_SI(g)
            case 'SI_2':
                lossgrad=SI(g,timescale=10,normpow=2).lossgrad
            case 'SI_3':
                lossgrad=SI(g,timescale=10,normpow=3).lossgrad
            case 'SI_4':
                lossgrad=SI(g,timescale=10,normpow=0,olpow=0).lossgrad

        P.lossgrads=[\
            #get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_),.001),\
            #get_barronweight(1.0,Barron._eval_),\
            #losses.get_lossgrad_NONSI(g),\
            lossgrad,\
            losses.get_lossgrad_SI(g),\
            losses.transform_grad(\
                lambda params,X,Y,rho: losses.norm(g(params,X),rho),\
                #T=lambda x:jnp.exp(jax.nn.relu(jnp.abs(jnp.log(x))-7.0)),\
                T=lambda x:jnp.log(x)**2,\
                #T=lambda x:abs(x)**(1/7),\
                fromrawloss=True)
            #losses.get_lossgrad_SI(P.learner._eval_)
        ]
        P.lossnames=['custom','SI','norm']

        #

        P.lossnames.append('second_derivative')
        P.lossgrads.append(Hloss(g).lossgrad)

        #

        P.lossweights=[1.0,0.0,1.0,0.0]
        P.info['learner']=P.learner.getinfo()
        P.name=mode
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
        super().plot(None,False)
    @staticmethod
    def getprofile(*a,**kw): return tracking.Profile()
    


def value_and_grad_mapped(f):
    f_single_x=lambda params,x:jnp.squeeze(f(params,jnp.expand_dims(x,axis=0)))
    return jax.vmap(jax.value_and_grad(f_single_x),in_axes=(None,0))



class SI:
    def __init__(self,f,timescale=100,normpow=3,olpow=0):
        self.grad_etc=jax.jit(lambda params,X,Y,rho: self.norm_and_overlap_and_grad(f,params,X,Y,1/rho))
        self.eps=1/timescale
        self.r=1-self.eps
        self.normpow=normpow
        self.olpow=olpow
        self.ff=None
        self.yf=None

    def lossgrad(self,params,X,Y,rho):
        normsq,overlap,G=self.grad_etc(params,X,Y,rho)
        self.ff=normsq if self.ff is None else normsq**self.eps*self.ff**self.r
        self.yf=overlap if self.yf is None else overlap**self.eps+self.r*self.yf**self.r
        #return 1-overlap**2/self.Z**2,tree_map(lambda g:g/self.Z**self.npow,G)
        return jnp.sqrt(self.ff),tree_map(lambda g:g*self.yf**self.olpow/self.ff**(self.normpow/2),G)

    @staticmethod
    def norm_and_overlap_and_grad(f,params,X,Y,weights):
        fX,GfX=value_and_grad_mapped(f)(params,X)
        normsq=jnp.average(fX**2*weights)
        overlap=jnp.average(fX*Y*weights)
        a=-normsq*Y+overlap*fX
        G=tree_map(lambda g:jnp.tensordot(a*weights,g,axes=(0,0)),GfX)
        return normsq,overlap,G


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