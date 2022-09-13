
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
    prepdisplay1(display,profile)

    genX0=lambda samples: profile._X0_distr_(tracking.nextkey(),samples,profile.n,profile.d)
    X0=genX0(profile.nrunners)

    sampler=sampling.Sampler(run.p,profile.proposalfn,X0)
    #sampler=sampling.Sampler(jax.jit(lambda X:profile.wavefunction(X)**2),profile.proposalfn,X0)
    burn(run,sampler)

    display.delkeys('cd1')
    prepdisplay2(display,profile)
    sample(run,sampler,returnsamples=False)



def burn(run,sampler):

    estimates100={name:tracking.RunningAvg(run.burn_avg_of) for name in run.observables.keys()}

    for i in range(run.maxburnsteps):
        run.trackcurrent('timeslices',i)
        for name,O in run.observables.items():

            sampler.step()

            newest=jnp.sum(run.qpratio(sampler.X)*O(sampler.X))/jnp.sum(run.qpratio(sampler.X))
            run.trackcurrent('estimate k '+name,estimates100[name].update(newest))

        run.display.draw()
        if act_on_input(tracking.checkforinput())=='b': break


def sample(run,sampler,returnsamples=True):
    independentnumsums={name:jnp.zeros((run.nrunners,)) for name in run.observables.keys()}
    independentdenomsums={name:jnp.zeros((run.nrunners,)) for name in run.observables.keys()}
    independentestimates=dict()

    samples=None # these will have dimensions (runners,time,n,d)
    timeslices=0

    estimatesforplot={name:[] for name in run.observables}

    for i in range(run.maxiterations):
        sampler.step()
        if i%run.thinningratio==0:
            timeslices+=1
            run.trackcurrent('timeslices',timeslices)
            run.trackcurrent('steps',i+1)

            if returnsamples:
                newsamples=jnp.expand_dims(sampler.X,axis=1)
                samples=jnp.concatenate([samples,newsamples],axis=1) if samples!=None else newsamples

            for name,O in run.observables.items():
                independentnumsums[name]+=run.qpratio(sampler.X)*O(sampler.X)
                independentdenomsums[name]+=run.qpratio(sampler.X)
                independentestimates[name]=independentnumsums[name]/independentdenomsums[name]

            confintervals={name:sampling.bootstrap_confinterval(\
                independentestimates[name],nresamples=500,q=jnp.array([2.5/100,97.5/100])) for name in run.observables}
            estimates={name:jnp.average(independentestimates[name]) for name in run.observables}

            for name in run.observables:
                run.trackcurrent('estimate '+name,estimates[name])
                run.trackcurrent('confinterval '+name,confintervals[name])
                estimatesforplot[name].append(estimates[name])

        run.trackcurrent('estimate '+name,estimates[name])
        run.display.draw()

        if act_on_input(tracking.checkforinput())=='b': break# and i>profile.minburnsteps: break

    for name,tv in zip(run.observables,run.trueenergies):
        fig,ax=plt.subplots()
        ax.plot(estimatesforplot[name])
        ax.plot([0,len(estimatesforplot[name])],2*[tv])
        ax.set_title('potential energy')
        ax.set_ylim(bottom=0,top=tv*1.2)
        fig.savefig('outputs/'+run.ID+'.pdf')


    return samples


            


def act_on_input(i):
    if i=='q': quit()
    return i

            

def prepdisplay1(display:disp.CompositeDisplay,run):
    cd,_=display.add(cdisplay.ConcreteDisplay(display.xlim,display.ylim),name='cd1')
    cd.add(disp.VSpace(3))
    cd.add(disp.FlexDisplay('timeslices',parse=lambda _,i:\
        'Burning samples. \n\nPress [b] to finish burning and begin registering samples.\n\n'+
        '{:,} steps and {:,} samples burned'.format(i,i*run.nrunners)))
    cd.add(disp.VSpace(5))

    for name,tv in zip(run.observables,run.trueenergies):

        cd.add(disp.FlexDisplay('estimate k '+name,parse=lambda _,x:\
            'Avg of last {:,} steps, {:,} samples:\n\n{:.4f}, relative error {:.2E}'.\
                format(x[1],x[1]*run.nrunners,x[0],jnp.log(x[0]/tv))))

        cd.add(disp.VSpace(3))

        transform=disp.R_to_I_formatter(tv,1.0)
        I=list(jnp.arange(tv-2.25,tv+2,.5)); L=['{:.2f}'.format(l) for l in I]
        cd.add(disp.Ticks(transform,I+[tv],L+['{:.2f} (true value)'.format(tv)]))
        cd.add(disp.Ticks(transform,I+[tv]))
        cd.add(disp.FlexDisplay('estimate k '+name,parse=\
            lambda D,x:transform(x[0],D.width)*textutil.dash+textutil.BOX+D.width*textutil.dash))
        cd.add(disp.Ticks(transform,I+[tv]))


def prepdisplay2(display:disp.CompositeDisplay,run):
    cd,_=display.add(cdisplay.ConcreteDisplay(display.xlim,display.ylim))
    cd.add(disp.VSpace(3))
    cd.add(disp.FlexDisplay('timeslices','steps',parse=lambda _,s,i:\
        'Saved {:,} samples from {:,} time slices, every {}th out of {} steps.'\
        .format(s*run.nrunners,s,run.thinningratio,i)))
    cd.add(disp.VSpace(3))

    for name,tv in zip(run.observables,run.trueenergies):

        cd.add(disp.FlexDisplay('estimate '+name,parse=lambda _,x:\
            'Estimate \n{:.4f}, relative error {:.2E}'.\
            format(x,jnp.log(x/tv))))
        cd.add(disp.VSpace(5))


        transform=disp.R_to_I_formatter(tv,0.2)
        def showinterval(display,t,I):
            T=lambda x: transform(x,displaywidth=display.width)
            interval=T(I[0])*' '+'['+(T(I[1])-T(I[0]))*textutil.dash+']'
            return textutil.overwrite(interval,T(t)*' '+textutil.BOX)
        cd.add(disp.FlexDisplay('estimate '+name,'confinterval '+name,parse=showinterval))
        I=list(jnp.arange(tv-.25,tv+.26,.1)); L=['{:.2f}'.format(l) for l in I]
        cd.add(disp.Ticks(transform,I+[tv]))
        cd.add(disp.Ticks(transform,I+[tv],L+['{:.2f} (true value)'.format(tv)]))
        cd.add(disp.Ticks(transform,I+[tv]))




class Run(tracking.Run):
    execprocess=execprocess
