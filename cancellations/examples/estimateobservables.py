
from cancellations.display import cdisplay
from ..utilities import tracking,arrayutil
from ..learning import sampling
from ..display import display as disp
import jax
import jax.random as rnd
import jax.numpy as jnp
import math



def getdefaultprofile():
    return tracking.Profile(\
        nrunners=1000,\
        _X0_distr_=lambda key,samples,n,d: rnd.normal(key,(samples,n,d)),\
        proposalfn=gaussianstepproposal(.1),\
        thinningratio=1,\
        observable=lambda X:jnp.sum(X**2,axis=(-2,-1)),\
        n=5,\
        d=1,\
        wavefunctions='not set',\
        densitynames='not set'
        )

def gaussianstepproposal(var):
    return lambda key,X: X+rnd.normal(key,X.shape)*math.sqrt(var)




def execprocess(run):
    profile,display=run,run.display
    prepdisplay(display,profile)

    genX0=lambda samples: profile._X0_distr_(tracking.nextkey(),samples,profile.n,profile.d)
    X0=genX0(profile.nrunners)

    samplers=[sampling.Sampler(jax.jit(lambda X:Psi(X)**2),profile.proposalfn,X0) for Psi in profile.wavefunctions]
    estimates={name:[] for name in profile.densitynames}

    while True:
        for name,sampler in zip(profile.densitynames,samplers):
            sampler.step()
            estimates[name].append(jnp.sum(profile.observable(sampler.X))/profile.nrunners)

            run.trackcurrent('estimate '+name,estimates[name][-1])

            display.draw()

            

def prepdisplay(display:disp.CompositeDisplay,profile):
    cd,_=display.add(cdisplay.ConcreteDisplay(display.xlim,display.ylim))
    for name in profile.densitynames:
        cd.add(disp.NumberPrint('estimate '+name))
        cd.add(disp.RplusBar('estimate '+name))





class Run(tracking.Run):
    execprocess=execprocess
