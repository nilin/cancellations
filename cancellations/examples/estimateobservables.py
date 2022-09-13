
# nilin


import profile
from cancellations.utilities import textutil
from ..display import cdisplay
from ..utilities import tracking,arrayutil,sampling,config as cfg
from ..display import display as disp
import jax
import jax.random as rnd
from collections import deque
import matplotlib.pyplot as plt
import jax.numpy as jnp
import math



def getdefaultprofile():
    return tracking.Profile(\
        name='sampling',\
        nrunners=1000,\
        _X0_distr_=lambda key,samples,n,d: rnd.normal(key,(samples,n,d)),\
        proposalfn=gaussianstepproposal(.1),\
        observables={'squarepotentialwell':lambda X:jnp.sum(X**2/2,axis=(-2,-1))},\
        n=5,\
        d=1,\
        p='not set',\
        qpratio='not set',\
        trueenergies='not set',\
        minburnsteps=100,\
        maxburnsteps=1000,\
        maxiterations=10000,\
        thinningratio=5,\
        burn_avg_of=100
        )

def gaussianstepproposal(var):
    return lambda key,X: X+rnd.normal(key,X.shape)*math.sqrt(var)



def execprocess(run):
    profile,display=run,run.display
    prepdisplay(display,profile)

    genX0=lambda samples: profile._X0_distr_(tracking.nextkey(),samples,profile.n,profile.d)
    X0=genX0(profile.nrunners)

    sampler=sampling.Sampler(run.p,profile.proposalfn,X0)
    sample(run,sampler)
    display.sd.pickdisplay('sd2'); display.fd.reset()
    sample(run,sampler,saveevery=run.thinningratio)



def sample(run,sampler,saveevery=None):

    estimates100={name:tracking.RunningAvg(run.burn_avg_of) for name in run.observables.keys()}
    Xs=[]

    for i in range(run.maxburnsteps):
        run.trackcurrent('steps',i)
        for name,O in run.observables.items():

            sampler.step()

            newest=jnp.sum(run.qpratio(sampler.X)*O(sampler.X))/jnp.sum(run.qpratio(sampler.X))
            run.trackcurrent('estimate k '+name,estimates100[name].update(newest))

        if saveevery!=None and i%saveevery==0:
            Xs.append(sampler.X)
            run.trackcurrent('timeslices',len(Xs))

        run.display.draw()
        if act_on_input(tracking.checkforinput())=='b': break

            


def act_on_input(i):
    if i=='q': quit()
    return i

            

def prepdisplay(display:disp.CompositeDisplay,run):
    cd,_=display.add(cdisplay.ConcreteDisplay(display.xlim,display.ylim),name='cd1')

    display.sd,_=cd.add(disp.SwitchDisplay())

    display.sd.add(disp.FlexDisplay('steps',parse=lambda _,i:\
        'Burning samples. \n\nPress [b] to finish burning and begin registering samples.\n\n'+
        '{:,} steps and {:,} samples burned'.format(i,i*run.nrunners)),name='sd1')

    display.sd.add(disp.FlexDisplay('steps','timeslices',parse=lambda _,s,sl:\
        'Saved {:,} samples from {:,} time slices, every {}th out of {} steps.'\
        .format(sl*run.nrunners,sl,run.thinningratio,s)),name='sd2')

    display.sd.pickdisplay('sd1')
    cd.add(disp.VSpace(5))

    for name,tv in zip(run.observables,run.trueenergies):

        display.fd,_=cd.add(disp.FlexDisplay('estimate k '+name,parse=lambda d,x:\
            'Avg of last {:,} steps, {:,} samples:\n\n{:.4f}, relative error {:.2E}'.\
                format(d.smoothers[0].actualk(),d.smoothers[0].actualk()*run.nrunners,x,jnp.log(x/tv))))

        cd.add(disp.VSpace(3))

        transform=disp.R_to_I_formatter(tv,1.0)
        I=list(jnp.arange(tv-2.25,tv+2,.5)); L=['{:.2f}'.format(l) for l in I]
        cd.add(disp.Ticks(transform,I+[tv],L+['{:.2f} (true value)'.format(tv)]))
        cd.add(disp.Ticks(transform,I+[tv]))
        cd.add(disp.FlexDisplay('estimate k '+name,parse=\
            lambda D,x:transform(x,D.width)*textutil.dash+textutil.BOX+D.width*textutil.dash))
        cd.add(disp.Ticks(transform,I+[tv]))




class Run(tracking.Run):
    execprocess=execprocess
