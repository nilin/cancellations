#
# nilin
# 
# 2022/7
#


from cancellations.examples import losses, Barronnorm as BN
import jax.numpy as jnp
from functools import partial
import jax
from cancellations.functions.symmetries import gen_Af
import jax.random as rnd
from cancellations.tracking import tracking
from jax.tree_util import tree_map
from cancellations.display import _display_

from cancellations.functions import _functions_, examplefunctions as examples, examplefunctions
from cancellations.functions._functions_ import Product
import sys

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


class ExpFit(runtemplate.Fixed_XY):
    processname='expfit'

    @classmethod
    def getprofile(cls,parentprocess,n,d,m,target,samples_train,minibatchsize):
        tracking.log('32 or 64: {}'.format(jnp.array([1.0]).dtype))

        P=profile=super().getprofile(n,d,samples_train,minibatchsize,target)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        P.learner=examplefunctions.ExpSlater(n=n,d=d,m=m)
        profile.lossgrads=[losses.get_lossgrad_SI(P.learner._eval_)]
        P.lossnames=['SI']
        P.lossweights=[100.0,0.01]
        P.name='{} n={} d={}'.format(P.learner.typename(),n,d)
        P.info={'target':target.getinfo(),'Ansatz':P.learner.getinfo()}
        return P

    def repeat(self,i):
        super().repeat(i)
        if i%1000==0:
            self.saveBnorm()
        if i%100==0:
            sysutil.save(self.profile.learner.compress(),os.path.join('outputs',tracking.sessionID,'learner'))

    def saveBnorm(self):
        epss=jnp.array(self.losses['SI'])
        i=len(epss)
        l=i//10
        self.eps=jnp.quantile(epss[-l:],.5)
        sysutil.write('{} | eps={}\n'.format(\
            self.profile.name,self.eps),\
            'outputs/expAnsatzloss.txt')

    def finish(self):
        self.saveBnorm()
        return self.eps
        #self.plot(self.profile)

    def plot(self,P):
        self.saveBnorm()
        fig,(ax1)=plt.subplots(1,1)
        plt.rcParams['text.usetex']
        fig.suptitle('Barron norm, n={}, m={}, {}'.format(P.n,P.m,P.ac))
        epss=jnp.array(self.losses['SI'])
        ax1.plot(epss,'r',label='$\epsilon$')
        ax1.axhline(y=self.eps,ls='--',color='r')
        ax1.set_yscale('log')
        ax1.legend()
        for path in ['plots',P.outpath_plot]:
            outpath=os.path.join(path,'expAnsatzloss_n={}_d={}___{}.pdf'.format(P.n,P.d,P.runID))
            sysutil.savefig(outpath)
            tracking.log(outpath)

class Run(ExpFit):

    @classmethod
    def getprofile(cls,parentprocess):
        n=parentprocess.browse(options=[1,2,3,4,5,6],displayoption=lambda o:'n={}'.format(o),msg='Select n')
        d=parentprocess.browse(options=[1,2,3],displayoption=lambda o:'d={}'.format(o),msg='Select d')
        m=parentprocess.browse(options=[2**k for k in range(6,12)],msg='Select Ansatz layer count')
        target=examplefunctions.QuadrantSlater(n=n,d=d)
        return ExpFit.getprofile(parentprocess,n,d,m,target,samples_train=10000,minibatchsize=100)

class ExpFitLoaded(ExpFit):
    @classmethod
    def getprofile(cls,parentprocess,X,Y,rho,m,minibatchsize):

        P=profile=runtemplate.Loaded_XY.getprofile(X,Y,rho,minibatchsize)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        P.learner=examplefunctions.ExpSlater(n=P.n,d=P.d,m=m)
        profile.lossgrads=[losses.get_lossgrad_SI(P.learner._eval_)]
        P.lossnames=['SI']
        P.lossweights=[100.0,0.01]
        P.name='{} n={} d={}'.format(P.learner.typename(),P.n,P.d)
        P.info={'Ansatz':P.learner.getinfo()}
        return P

### batch runs ###

class Runthrough(runtemplate.Run):
    def execprocess(self):
        if cfg.istest:
            n=4
            d=2
            minibatchsize=5
            its=250
        else:
            n=self.browse(options=[1,2,3,4,5,6],displayoption=lambda o:'n={}'.format(o),msg='Select n')
            d=self.browse(options=[1,2,3],displayoption=lambda o:'d={}'.format(o),msg='Select d')
            minibatchsize=100 #self.browse(options=[100,50,25,10,250],displayoption=lambda o:'minibatchsize={}'.format(o),msg='Select minibatchsize')
            its=self.browse(options=[10**4,5000,2500,1000],displayoption=lambda o:'{} iterations'.format(o),msg='Select iterations')

        X=rnd.uniform(rnd.PRNGKey(0),(10**5,n,d))
        rho=jnp.ones((X.shape[0],))/2**(n*d)

        explosses=[]
        Barronnorms=[]
        Barroneps=[]
        t_s=[]

        t_=jnp.arange(.5,2,.2)
        s_=jnp.arange(-1,1,.2)
        m_e=4096
        m_B=1024

        if cfg.istest:
            t_=[.5]
            s_=[-1]
            m_e=64
            m_B=32

        folder='batchoutputs'

        def save_some(**kw):
            out=dict(\
                explosses=explosses,\
                Barronnorms=Barronnorms,\
                Barroneps=Barroneps,\
                t_s=t_s)
            out.update(kw)
            sysutil.save(out,\
                path=os.path.join(folder,tracking.sessionID,'compare_n={}_d={}___{}'.format(n,d,tracking.sessionID)))

        if cfg.dump:
            sysutil.savewhatyoucan([globals(),locals()],os.path.join(folder,tracking.sessionID,'datadump0'))

        for t in t_:
            for s in s_:
                t_s.append((t,s))
                Y=examplefunctions.get_harmonic_oscillator2d(n,d).eval(t*X-s)
                Y=Y/losses.norm(Y,rho)

                BP=BN.BarronLossLoaded.getprofile(self,X,Y,rho,m_B,minibatchsize=minibatchsize).butwith(iterations=its)
                Barronnorm,eps=self.subprocess(BN.Run(profile=BP))
                Barronnorms.append(Barronnorm)
                Barroneps.append(eps)

                EP=ExpFitLoaded.getprofile(self,X,Y,rho,m_e,minibatchsize=minibatchsize).butwith(iterations=its)
                explosses.append(self.subprocess(ExpFit(profile=EP)))

                save_some(E_info=EP.learner.getinfo(),B_info=BP.learner.getinfo())

        if cfg.dump:
            sysutil.savewhatyoucan(locals(),os.path.join(folder,tracking.sessionID,'datadump1'))


#        for m in ms:
#            P=ExpFitLoaded.getprofile(self,X,Y,rho,m,minibatchsize=100).butwith(iterations=10000)
#            eps=self.subprocess(ExpFit(profile=P))
#            losses.append(eps)

#        bP=BN.BarronLossLoaded.getprofile(self,X,Y,rho,1024).butwith(iterations=10000)
#        Barronnorm,eps=self.subprocess(BN.Run(profile=bP))
#
#        fig,(ax1)=plt.subplots(1,1)
#        plt.rcParams['text.usetex']
#        fig.suptitle('Normalized loss, n={}, d={}'.format(n,d))
#        ax1.plot(ms,losses,'bo-',label='$\epsilon$')
#        if plotBarron and eps<.1:
#        #if plotBarron:
#            ax1.plot(ms,Barronnorm**2/ms,'r:',label='$|\psi|_A^2/m$')
#        ax1.set_xscale('log')
#        ax1.set_yscale('log')
#        ax1.legend()
#        outpath=os.path.join('plots','expAnsatzlosses_n={}_{}___{}.pdf'.format(n,d,tracking.sessionID))
#        outpath_data=os.path.join('plots','expAnsatzlosses_n={}_{}___{}'.format(n,d,tracking.sessionID))
#        sysutil.savefig(outpath)
#        sysutil.showfile(outpath)
#        #sysutil.save([ms,losses,target.getinfo()],outpath_data)
        

    @classmethod
    def getprofile(cls,parentprocess): return tracking.dotdict()
