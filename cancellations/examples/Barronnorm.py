#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses
import jax.numpy as jnp
from functools import partial
import jax
from cancellations.tracking import tracking
from jax.tree_util import tree_map
from cancellations.display import _display_

from cancellations.functions import _functions_, examplefunctions as examples
from cancellations.functions._functions_ import Product

from cancellations.utilities.permutations import allpermtuples
from cancellations.tracking.browse import Browse
import math
from cancellations.tracking import runconfig as cfg
from cancellations.run import runtemplate
from cancellations.tracking.tracking import dotdict, log, sysutil
import matplotlib.pyplot as plt
import re
import os


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################



class BarronLoss(runtemplate.Fixed_XY):
    processname='Barron_norm'

    @classmethod
    def getprofile(cls,parentprocess,target,n,d,m,delta=1/10.0**4,minibatchsize=100):

        samples_train=10**5
        P=profile=super().getprofile(n,d,samples_train,minibatchsize,target)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        ac='softplus'
        P.mode='ANTISYM'

        match P.mode:
            case 'ANTISYM': Ansatz=_functions_.ASBarron(n=n,d=d,m=m,ac=ac)
            case 'RAW': Ansatz=_functions_.Barron(n=n,d=d,m=m,ac=ac)
        P.learner=Ansatz
        profile.lossgrads=[\
            cls.get_threshold_lg(losses.get_lossgrad_SI(Ansatz._eval_),delta),\
            cls.get_learnerweightnorm(1.0,Ansatz._eval_),\
        ]
        P.lossnames=['eps','Barron norm estimate']
        P.lossweights=[100.0,0.01]
        P.name='{} n={} d={}'.format(P.learner.typename(),n,d)
        P.info={'target':target.getinfo(),'Barron':Ansatz.getinfo()}
        return P

    def repeat(self,i):
        super().repeat(i)
        if i%1000==0:
            self.saveBnorm()
        if i%1000==0:
            sysutil.save(self.profile.learner.compress(),os.path.join('outputs',tracking.sessionID,'learner'))

    def saveBnorm(self):
        Bnorms=jnp.array(self.losses['Barron norm estimate'])
        epss=jnp.array(self.losses['eps'])
        i=len(epss)
        l=i//10
        self.Bnorm=jnp.quantile(Bnorms[-l:],.5)
        self.eps=jnp.quantile(epss[-l:],.5)
        sysutil.write('{} | Barron={} | eps={}\n'.format(\
            self.profile.name,self.Bnorm,self.eps),\
            'outputs/Barronnorms.txt')

    def finish(self):
        self.saveBnorm()
        #self.plot(self.profile)
        return (self.Bnorm,self.eps)

    def plot(self,P):

        self.saveBnorm()
        fig,(ax0,ax1)=plt.subplots(2,1)
        plt.rcParams['text.usetex']
        fig.suptitle('{} Barron norm, n={}, m={}, {}'.format(P.mode,P.n,P.m,P.ac))
        Bnorms=jnp.array(self.losses['Barron norm estimate'])
        ax0.plot(Bnorms,'b',label='Barron Ansatz weight norm')
        ax0.set_ylim(0,self.Bnorm*3)
        ax0.legend()
        epss=jnp.array(self.losses['eps'])
        ax1.plot(epss,'r',label='$\epsilon$')
        ax1.axhline(y=self.eps,ls='--',color='r')
        if self.eps<.01:
            ax0.axhline(y=self.Bnorm,ls='--',color='b',label='$\epsilon$-smooth Barron norm estimate')
        ax1.set_yscale('log')
        ax1.legend()

        if cfg.debug:
            breakpoint()
        for path in ['plots',P.outpath_plot]:
            outpath=os.path.join(path,'Bnorm_n={}_d={}___{}.pdf'.format(P.n,P.d,P.runID))
            sysutil.savefig(outpath)
        #sysutil.showfile(outpath)

    @staticmethod
    def get_learnerweightnorm(p,f):
        def loss(p,prodparams):
            #_,params=prodparams
            params=prodparams
            (W,b),a=params
            w1=jnp.squeeze(abs(a))
            w2=jnp.sum(jnp.abs(W),axis=(-2,-1))+jnp.abs(b)
            assert(w1.shape==w2.shape)
            if p==float('inf'):
                return jnp.max(w1*w2)
            else:
                return jnp.sum((w1*w2)**p)**(1/p)
        lossfn=lambda params,X,Y,rho: loss(p,params)/losses.norm(f(params,X),rho)
        return jax.jit(jax.value_and_grad(lossfn))

    @staticmethod
    def get_threshold_lg(lg,delta,**kw):
        delta10ths=delta/10.0
        def _eval_(params,*X):
            val,grad=lg(params,*X)
            weight=jax.nn.sigmoid(val/delta10ths-10.0)
            return val,tree_map(lambda A:weight*A,grad)
        return jax.jit(_eval_)

class Run(BarronLoss):
    processname='Barron_norm'

    @classmethod
    def getprofile(cls, parentprocess,n=None,d=None,m=None,target=None):
        if n is None:
            n=parentprocess.browse(options=[1,2,3,4,5,6],displayoption=lambda o:'n={}'.format(o),msg='Select n')
        if d is None:
            d=parentprocess.browse(options=[1,2,3],displayoption=lambda o:'d={}'.format(o),msg='Select d')
        if m is None:
            m=parentprocess.browse(options=[2**k for k in range(6,12)],msg='Select Ansatz layer count')
        if target is None:
            #target=examples.QuadrantSlater(n=n,d=d)
            target=examples.get_harmonic_oscillator2d(n,d)
        P=super().getprofile(parentprocess, target, n, d, m, minibatchsize=10)
        return P
    
class BarronLossLoaded(BarronLoss):
    @classmethod
    def getprofile(cls,parentprocess,X,Y,rho,m,delta=1/10.0**4,minibatchsize=100):

        P=profile=runtemplate.Loaded_XY.getprofile(X,Y,rho,minibatchsize)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        ac='softplus'

        Ansatz=_functions_.ASBarron(n=P.n,d=P.d,m=m,ac=ac)
        P.learner=Ansatz
        profile.lossgrads=[\
            cls.get_threshold_lg(losses.get_lossgrad_SI(Ansatz._eval_),delta),\
            cls.get_learnerweightnorm(1.0,Ansatz._eval_),\
        ]
        P.lossnames=['eps','Barron norm estimate']
        P.lossweights=[100.0,0.01]
        P.name='{} n={} d={}'.format(P.learner.typename(),P.n,P.d)
        P.info={'Barron':Ansatz.getinfo()}
        return P

### plots ###

class Plot(Run):
    def execprocess(self):
        lines=sysutil.read('outputs/Barronnorms.txt')
        modes=[[m for m in ['ANTISYM','RAW'] if m in l][0] for l in lines]
        ns=[re.findall('n=(.)',l)[0] for l in lines]
        ds=[re.findall('d=(.)',l)[0] for l in lines]
        Barrons=[float(re.findall('Barron=([^\s]*)',l)[0]) for l in lines]
        epss=[float(re.findall('eps=([^\s]*)',l)[0]) for l in lines]

        delta=self.browse(options=[.5,1/10,1/100,1/1000,1/10000],msg='Pick threshold')
        d_=self.browse(options=[1,2,3],msg='Pick dimension d to plot')

        stats=dict()
        for mode,n,d,Barron,eps in zip(modes,ns,ds,Barrons,epss):
            if float(eps)>delta or int(d)!=d_:
                continue
            if (mode,n) not in stats:
                stats[(mode,n)]=[]
            stats[(mode,n)].append(float(Barron))

        minstats={key:min(Bs) for key,Bs in stats.items()}

        for mode_,c in zip(['RAW','ANTISYM'],['r','b']):
            stats_={int(n):B for (mode,n),B in minstats.items() if mode==mode_}
            plt.scatter(list(stats_.keys()),list(stats_.values()),color=c)
            plt.plot(list(stats_.keys()),list(stats_.values()),color=c)
            plt.yscale('log')

        outpath='plots/Barronplot.pdf'
        sysutil.savefig(outpath)
        sysutil.showfile(outpath)

    @staticmethod
    def getprofile(*a,**kw): return tracking.dotdict()

### plots ###

### batch runs ###

class Runthrough(_display_.Process):
    def execprocess(self):
        P=self.profile
        mode=self.subprocess(Browse(options=['ANTISYM','RAW']))

        for n in P.ns:
            if mode=='ANTISYM':
                m=1024
                P2=Run.getprofile(self,n=n,d=P.d,m=m,mode='ANTISYM',delta=.01).butwith(iterations=10**4)
                self.subprocess(Run(profile=P2))

            if mode=='RAW':
                m=32
                P1=Run.getprofile(self,n=n,d=P.d,m=m,mode='ANTISYM').butwith(iterations=10**4)
                self.subprocess(Run(profile=P1))

                Barron=Run.get_Ansatz(P1.butwith(m=math.factorial(n)*m))
                Barron.weights=self.expandweights(P1.learner.weights)

                P4=Run.getprofile(self,n=n,d=P.d,m=m,mode='RAW',Ansatz=Barron).butwith(iterations=10**4)
                self.subprocess(Run(profile=P4))


    # m blocks of n! permutations 
    @staticmethod
    def expandweights(params):

        (W,b),a=params
        m,n,d=W.shape
        perms,signs=allpermtuples(n)
        
        W_=W[:,perms,:].reshape(-1,n,d)
        b_=jnp.repeat(b,len(signs))
        a_=jnp.kron(a,signs)

        return [(W_,b_),a_]

    @classmethod
    def getprofile(cls,parentprocess):
        P=tracking.dotdict()
        #P.d=parentprocess.browse(options=[1,2,3],msg='Pick d')
        P.d=3
        n_options=list(range(1,9))
        P.ns=parentprocess.browse(options=n_options,optionstrings=['n={}'.format(n) for n in n_options],\
            onlyone=False,msg='Pick ns')
        return P
