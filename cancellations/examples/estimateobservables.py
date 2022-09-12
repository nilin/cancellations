
# nilin


import profile
from cancellations.utilities import textutil
from ..display import cdisplay
from ..utilities import tracking,arrayutil,sampling
from ..display import display as disp
import jax
import jax.random as rnd
from collections import deque
import jax.numpy as jnp
import math



def getdefaultprofile():
    return tracking.Profile(\
        nrunners=1000,\
        _X0_distr_=lambda key,samples,n,d: rnd.normal(key,(samples,n,d)),\
        proposalfn=gaussianstepproposal(.1),\
        observables={'squarepotentialwell':lambda X:jnp.sum(X**2/2,axis=(-2,-1))},\
        n=5,\
        d=1,\
        wavefunction='not set',\
        minburnsteps=1000,\
        maxburnsteps=5000,\
        maxiterations=100000,\
        thinningratio=10
        )

def gaussianstepproposal(var):
    return lambda key,X: X+rnd.normal(key,X.shape)*math.sqrt(var)



def execprocess(run):
    profile,display=run,run.display
    prepdisplay1(display,profile)

    genX0=lambda samples: profile._X0_distr_(tracking.nextkey(),samples,profile.n,profile.d)
    X0=genX0(profile.nrunners)

    sampler=sampling.Sampler(jax.jit(lambda X:profile.wavefunction(X)**2),profile.proposalfn,X0)
    burn(run,sampler)

    display.delkeys('cd1')
    prepdisplay2(display,profile)
    sample(run,sampler,returnsamples=False)



def burn(run,sampler):

    estimates10={name:tracking.RunningAvg(1000) for name in run.observables.keys()}

    for i in range(run.maxburnsteps):
        run.trackcurrent('timeslices',i)
        for name,O in run.observables.items():

            sampler.step()

            newest=jnp.average(O(sampler.X))
            run.trackcurrent('estimate 1000 '+name,estimates10[name].update(newest))

        run.display.draw()
        if act_on_input(tracking.checkforinput())=='b': break


def sample(run,sampler,returnsamples=True):
    independentsums={name:jnp.zeros((run.nrunners,)) for name in run.observables.keys()}
    independentestimates=dict()

    samples=None # these will have dimensions (runners,time,n,d)
    timeslices=0

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
                independentsums[name]+=O(sampler.X)
                independentestimates[name]=independentsums[name]/timeslices

            confintervals={name:sampling.bootstrap_confinterval(\
                independentestimates[name],nresamples=250,q=jnp.array([.01,.99])) for name in run.observables}
            estimates={name:jnp.average(independentestimates[name]) for name in run.observables}

            for name in run.observables:
                run.trackcurrent('estimate '+name,estimates[name])
                run.trackcurrent('confinterval '+name,confintervals[name])

        run.trackcurrent('estimate '+name,estimates[name])
        run.display.draw()

        if act_on_input(tracking.checkforinput())=='b' and i>profile.minburnsteps: break

    return samples


            


def act_on_input(i):
    if i=='q': quit()
    return i

            

def prepdisplay1(display:disp.CompositeDisplay,run):
    cd,_=display.add(cdisplay.ConcreteDisplay(display.xlim,display.ylim),name='cd1')
    cd.add(disp.VSpace(3))
    cd.add(disp.FlexDisplay('timeslices',parse=lambda _,i:\
        'Burning samples. \n\nPress [b] to finish burning and begin registering samples.\n\n'+
        '{} steps and {} samples burned'.format(i,i*run.nrunners)))
    cd.add(disp.VSpace(5))

    for name in run.observables:
        tv=6.25

        cd.add(disp.FlexDisplay('estimate 1000 '+name,parse=lambda _,x:\
            'Avg of last {:,} steps, {:,} samples:\n\n{:.4f}, relative error {:.1%}'.\
                format(x[1],x[1]*run.nrunners,x[0],jnp.log(x[0]/tv))))

        cd.add(disp.VSpace(3))

        transform=disp.R_to_I_formatter(tv,1.0)
        I=list(jnp.arange(5,8.5,.5)); L=['{:.2}'.format(l) for l in I]
        cd.add(disp.Ticks(transform,I+[tv],L+['6.25 (true value)']))
        cd.add(disp.Ticks(transform,I+[tv]))
        cd.add(disp.FlexDisplay('estimate 1000 '+name,parse=\
            lambda D,x:transform(x[0],D.width)*textutil.dash+textutil.BOX+D.width*textutil.dash))
        cd.add(disp.Ticks(transform,I+[tv]))


def prepdisplay2(display:disp.CompositeDisplay,run):
    cd,_=display.add(cdisplay.ConcreteDisplay(display.xlim,display.ylim))
    cd.add(disp.VSpace(3))
    cd.add(disp.FlexDisplay('timeslices','steps',parse=lambda _,s,i:\
        'Saved {} samples from {} time slices, every {}th out of {} steps.'\
        .format(s*run.nrunners,s,run.thinningratio,i)))
    cd.add(disp.VSpace(3))

    for name in run.observables:
        tv=6.25

        cd.add(disp.FlexDisplay('estimate '+name,parse=lambda _,x:\
            'Estimate \n{:.4f}, relative error {:.2%}'.\
            format(x,jnp.log(x/tv))))
        cd.add(disp.VSpace(5))


        transform=disp.R_to_I_formatter(tv,0.2)
        def showinterval(display,I):
            T=lambda x: transform(x,displaywidth=display.width)
            return T(I[0])*' '+'['+(T(I[1])-T(I[0]))*textutil.dash+']'
        cd.add(disp.FlexDisplay('confinterval '+name,parse=showinterval))
        I=list(jnp.arange(6,6.51,.1)); L=['{:.2}'.format(l) for l in I]
        cd.add(disp.Ticks(transform,I+[tv]))
        cd.add(disp.Ticks(transform,I+[tv],L+['6.25 (true value)']))
        cd.add(disp.Ticks(transform,I+[tv]))




class Run(tracking.Run):
    execprocess=execprocess
