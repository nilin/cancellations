#
# nilin
# 
# 2022/7
#


from cancellations.examples import Ansatznorms, losses
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


class BarronLossLoaded(Ansatznorms.BarronLoss):
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


class ExpFitLoaded(Ansatznorms.ExpFit):
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


class General(ExpFitLoaded):
    @classmethod
    def getprofile(cls,parentprocess,X,Y,rho,m,minibatchsize):

        P=profile=runtemplate.Loaded_XY.getprofile(X,Y,rho,minibatchsize)
        P.Y=P.Y/losses.norm(P.Y[:1000],P.rho[:1000])
        cls.initsampler(P)

        P.learner=examplefunctions.getlearner_example(P.n,P.d)
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
            general=True
            skipBarron=True
        else:
            n=self.browse(options=[1,2,3,4,5,6],displayoption=lambda o:'n={}'.format(o),msg='Select n')
            d=self.browse(options=[1,2,3],displayoption=lambda o:'d={}'.format(o),msg='Select d')
            minibatchsize=100 #self.browse(options=[100,50,25,10,250],displayoption=lambda o:'minibatchsize={}'.format(o),msg='Select minibatchsize')
            its=self.browse(options=[10**4,5000,2500,1000],displayoption=lambda o:'{} iterations'.format(o),msg='Select iterations')
            general=self.browse(options=[True,False],msg='general or not')
            skipBarron=True


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
                t_s=t_s,\
                general=general)
            out.update(kw)
            sysutil.save(out,\
                path=os.path.join(folder,tracking.sessionID,'compare_n={}_d={}___{}'.format(n,d,tracking.sessionID)))

        #sysutil.save(out,\
        #    path=os.path.join(folder,tracking.sessionID,'compare_n={}_d={}___{}'.format(n,d,tracking.sessionID)))

        if cfg.dump:
            sysutil.savewhatyoucan([globals(),locals()],os.path.join(folder,tracking.sessionID,'datadump0'))

        done=0
        for t in t_:
            for s in s_:
                t_s.append((t,s))
                Y=examplefunctions.get_harmonic_oscillator2d(n,d).eval(t*X-s)
                Y=Y/losses.norm(Y,rho)

                if not skipBarron:
                    BP=BarronLossLoaded.getprofile(self,X,Y,rho,m_B,minibatchsize=minibatchsize).butwith(iterations=its)
                    Barronnorm,eps=self.subprocess(Ansatznorms.Run(profile=BP))
                    Barronnorms.append(Barronnorm)
                    Barroneps.append(eps)

                if general:
                    EP=General.getprofile(self,X,Y,rho,m_e,minibatchsize=minibatchsize).butwith(iterations=its)
                else:
                    EP=ExpFitLoaded.getprofile(self,X,Y,rho,m_e,minibatchsize=minibatchsize).butwith(iterations=its)

                explosses.append(self.subprocess(Ansatznorms.ExpFit(profile=EP)))

                if skipBarron: save_some(E_info=EP.learner.getinfo(),B_info=BP.learner.getinfo())
                else: save_some(E_info=EP.learner.getinfo())

                done+=1
                tracking.log('{}/{} done'.format(done,len(t_)*len(s_)))

        if cfg.dump:
            sysutil.savewhatyoucan(locals(),os.path.join(folder,tracking.sessionID,'datadump1'))

        

    @classmethod
    def getprofile(cls,parentprocess): return tracking.dotdict()
