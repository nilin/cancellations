
# nilin


import profile
from re import I
from cancellations.utilities import textutil,sysutil
from ..display import cdisplay
from ..utilities import numutil, tracking,sampling,config as cfg
from ..display import display as disp
import jax
import jax.random as rnd
from collections import deque
import matplotlib.pyplot as plt
import jax.numpy as jnp
import copy
import math






class Run(cdisplay.Run):

    exname='estimateobservables'

    def execprocess(run):
        profile,display=run,run.display

        genX0=lambda samples: profile._X0_distr_(tracking.nextkey(),samples,profile.n,profile.d)
        X0=genX0(profile.nrunners)

        sampler=sampling.Sampler(run.p,profile.proposalfn,X0) if run.sampler==None else run.sampler

        Xblock=[]
        run.obsestimator=ObsEstimator(run.observables,run.qpratio,run.preburnt)
        run.obsestimates=dict()

        for i in range(run.maxiterations):
            X=sampler.step()

            if run.blocksize!=None and i%(run.thinningratio*run.blocksize)==0 and i!=0:
                sysutil.save(Xblock,run.outpath+'block {}-{}'.format(i//run.thinningratio-len(Xblock),i//run.thinningratio))
                sysutil.write('{} slices'.format(i//run.thinningratio),run.outpath+'metadata.txt',mode='w')
                Xblock=[]

            if run.thinningratio!=None and i%run.thinningratio==0: Xblock.append(X)

            if i%run.estevery==0:
                numutil.appendtoeach(run.obsestimates,run.obsestimator.update(X))
                for obs,est in run.obsestimator.estimates().items():
                    run.trackcurrent('estimate '+obs,est)

            run.trackcurrent('steps',i)
            run.trackcurrent('timeslices',i//run.thinningratio)
            run.display.draw()
            if act_on_input(tracking.checkforinput(),run)=='b': break


    def prepdisplay(run):
        display=run.display
        cd,_=display.add(cdisplay.ConcreteStackedDisplay(display.xlim,display.ylim))

        cd.add(disp.FlexDisplay('steps','timeslices',parse=lambda _,s,sl:\
            'Saved {:,} samples from {:,} time slices, every {}th out of {} steps.'\
            .format(sl*run.nrunners,sl,run.thinningratio,s)))

        cd.add(disp.VSpace(5))

        def addline(obs):
            tracking.log(obs)
            tv=run.trueenergies[obs]

            cd.add(disp.VSpace(3))
            cd.add(disp.FlexDisplay('estimate '+obs,parse=\
                lambda d,x:'estimate {} {:.4f}, relative error {:.2E}'.format(obs,x,jnp.log(x/tv))))

            cd.add(disp.VSpace(1))

            T=lambda t: disp.R_to_I_formatter(tv,1)(t,display.width)
            class Range(disp.DynamicRange):
                def gettransform(self): self.center=tv; self.rangewidth=1; self.T=T

            cd.add(Range(run.getqueryfn('estimate '+obs),customticks=[tv],customlabels='true value'))

        for obs in run.observables.keys():
            addline(obs)

    @staticmethod
    def getdefaultprofile():
        return tracking.Profile(\
            name='sampling',\
            nrunners=1000,\
            _X0_distr_=lambda key,samples,n,d: rnd.normal(key,(samples,n,d)),\
            proposalfn=gaussianstepproposal(.1),\
            observables={'V':lambda X:jnp.sum(X**2/2,axis=(-2,-1))},\
            n=5,\
            d=1,\
            sampler=None,\
            preburnt=False,\
            p='not set',\
            qpratio='not set',\
            trueenergies={'V':6.25},\
            minburnsteps=100,\
            maxburnsteps=2000,\
            maxiterations=10000,\
            thinningratio=5,\
            blocksize=1000,\
            estevery=10,\
            )


def gaussianstepproposal(var):
    return lambda key,X: X+rnd.normal(key,X.shape)*math.sqrt(var)

class ObsEstimator:
    def __init__(self,observables,qpratio,preburnt=False,**kw):
        self.qpratio=qpratio
        self.observables=observables

        RunningAvgClass=tracking.InfiniteRunningAvg if preburnt else tracking.ExpRunningAverage
        self.estimators={name:RunningAvgClass(**kw) for name in observables.keys()}
        self.denomestimator=RunningAvgClass(**kw)
#        self.estimators={name:tracking.ExpRunningAverage(**kw) for name in observables.keys()}
#        self.denomestimator=tracking.ExpRunningAverage(**kw)

    def update(self,X):
        for name,O in self.observables.items():
            self.estimators[name].update(jnp.sum(self.qpratio(X)*O(X)))
            self.denomestimator.update(jnp.sum(self.qpratio(X)))
            pass
        return self.estimates()
        
    def estimates(self):
        denomavg=self.denomestimator.avg()
        return {name:numutil.trycomp(lambda x,y:x/y,num.avg(),denomavg) for name,num in self.estimators.items()}


def act_on_input(i,run):
    if i=='q': quit()
    if i=='o': sysutil.showfile(tracking.currentprocess().outpath)
    if i=='p': plot(run,run.obsestimates)
    return i

def plot(run,ests):
    fig,ax=plt.subplots()
    for name in run.observables.keys():
        tv,est=run.trueenergies[name],ests[name]
        ax.plot(est,'b:')
        ax.plot([0,len(est)],[tv,tv])
        ax.set_ylim([0,tv*1.2])
    sysutil.savefig(run.outpath+'plot.pdf',fig=fig)


