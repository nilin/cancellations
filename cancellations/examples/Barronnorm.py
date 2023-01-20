#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses
import jax.numpy as jnp
from functools import partial
import jax
from cancellations.config import tracking
from jax.tree_util import tree_map
from cancellations.display import _display_

from cancellations.functions import _functions_, examplefunctions as examples
from cancellations.functions._functions_ import Product

import math
from cancellations.config import config as cfg
from cancellations.run import runtemplate
from cancellations.config.tracking import Profile, log, sysutil
import matplotlib.pyplot as plt
import re
import os


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################


def getBarronfn(P):
    match P.mode:
        case 'ANTISYM':
            return Product(_functions_.IsoGaussian(1.0),\
                _functions_.ASBarron(n=P.n,d=P.d,m=P.m,ac=P.ac))
        case 'RAW':
            return Product(_functions_.IsoGaussian(1.0),\
                _functions_.Barron(n=P.n,d=P.d,m=P.m,ac=P.ac))


class Run(runtemplate.Fixed_XY):
    processname='Barron_norm'

    @classmethod
    def getprofile(cls,parentprocess,**kw):

        n=parentprocess.browse(options=[1,2,3,4,5,6],msg='Select n')\
            if 'n' not in kw else kw['n']
        d=parentprocess.browse(options=[1,2,3],msg='Select d')\
            if 'd' not in kw else kw['d']
        m=parentprocess.browse(options=[2**k for k in range(6,12)],msg='Select Barron layer count')\
            if 'm' not in kw else kw['m']
        mode=parentprocess.browse(options=['ANTISYM','RAW'],msg='Select Barron norm type')\
            if 'mode' not in kw else kw['mode']

        samples_train=10**5
        minibatchsize=100

        if 'imax' in kw:
            target=examples.get_harmonic_oscillator2d(n,d,imax=kw['imax'])
        else:
            target=examples.get_harmonic_oscillator2d(n,d)

        P=profile=super().getprofile(n,d,samples_train,minibatchsize,target).butwith(m=m,mode=mode)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        # temporary test
        x_,y_,r_=P.sampler.sample()
        log(losses.norm(y_,r_))
        # temporary test

        P.ac='softplus'
        Barron=getBarronfn(P)
        P.learner=Barron
        profile.lossgrads=[\
            get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_),1/10.0**4),\
            get_barronweight(1.0,Barron._eval_),\
        ]
        P.lossnames=['eps','Barron norm estimate']
        P.lossweights=[100.0,0.01]
        P.name='{} n={} d={}'.format(P.mode,n,d)
        P.info={'target':target.getinfo(),'Barron':Barron.getinfo()}
        return P

    def repeat(self,i):
        super().repeat(i)
        if i%1000==0:
            self.saveBnorm()

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
        self.plot(self.profile)

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
        outpath=os.path.join('plots','Bnorm_{}_n={}_{}___{}.pdf'.format(P.mode,P.n,P.ac,tracking.sessionID))
        sysutil.savefig(outpath)
        sysutil.showfile(outpath)


### plots ###

class Plot(Run):
    def execprocess(self):
        lines=sysutil.read('outputs/Barronnorms.txt')
        modes=[[m for m in ['ANTISYM','RAW'] if m in l][0] for l in lines]
        ns=[re.findall('n=(.)',l)[0] for l in lines]
        ds=[re.findall('d=(.)',l)[0] for l in lines]
        Barrons=[float(re.findall('Barron=([^\s]*)',l)[0]) for l in lines]
        epss=[float(re.findall('eps=([^\s]*)',l)[0]) for l in lines]

        delta=self.browse(options=[1/10,1/100,1/1000,1/10000],msg='Pick threshold')
        d_=self.browse(options=[1,2,3],msg='Pick dimension d to plot')

        stats=dict()
        for mode,n,d,Barron,eps in zip(modes,ns,ds,Barrons,epss):
            if float(eps)>delta or int(d)!=d_:
                continue
            if (mode,n) not in stats:
                stats[(mode,n)]=[]
            stats[(mode,n)].append(float(Barron))

        minstats={key:min(Bs) for key,Bs in stats.items()}

        for mode_,c in zip(['ANTISYM','RAW'],['b','r']):
            stats_={int(n):B for (mode,n),B in minstats.items() if mode==mode_}
            plt.scatter(list(stats_.keys()),list(stats_.values()),color=c)
            plt.yscale('log')

        outpath='plots/Barronplot.pdf'
        sysutil.savefig(outpath)
        sysutil.showfile(outpath)

    @staticmethod
    def getprofile(*a,**kw): return tracking.Profile()

### plots ###

### loss functions ###

def get_barronweight(p,f):
    def loss(p,prodparams):
        _,params=prodparams
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

#def get_threshold_lg(lg,delta,**kw):
#    return losses.transform_grad(losses.get_lossgrad_SI(lg,**kw),lambda l: jax.nn.softplus(l/delta-1)*delta)

def get_threshold_lg(lg,delta,**kw):
    delta10ths=delta/10.0
    def _eval_(params,*X):
        val,grad=lg(params,*X)
        weight=jax.nn.sigmoid(val/delta10ths-10.0)
        return val,tree_map(lambda A:weight*A,grad)
    return jax.jit(_eval_)

### loss functions ###

### batch runs ###

class Runthrough(_display_.Process):
    def execprocess(self):
        P=self.profile
        for n in range(1,P.nmax+1):
            for mode in ['ANTISYM','RAW']:
                try:
                    P=Run.getprofile(self,n=n,d=P.d,m=1024,mode=mode,imax=P.nmax).butwith(iterations=10**4)
                    self.subprocess(Run(profile=P))
                except Exception as e:
                    tracking.log(str(e))

    @classmethod
    def getprofile(cls,parentprocess):
        P=tracking.Profile()
        P.d=parentprocess.browse(options=[1,2,3],msg='Pick d')
        P.nmax=parentprocess.browse(options=[4,5,6],msg='Pick nmax')
        return P
