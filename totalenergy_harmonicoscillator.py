from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d,estimateobservables
from cancellations.functions import examplefunctions as ef
from cancellations.display import cdisplay
from cancellations.utilities import tracking, arrayutil, browse, batchjob, energy
import jax.numpy as jnp
import jax
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')




psi=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).f
E_kin_local=arrayutil.noparams(energy.genlocalkinetic(psi))
psi=arrayutil.noparams(psi)

batch.name1='Total energy K+V for true ground state'
batch.task1=estimateobservables.Run
batch.genprofile1=lambda _: estimateobservables.getdefaultprofile().butwith(\
    p=lambda X:psi(X)**2,\
    qpratio=lambda X:jnp.ones(X.shape[0],),\
    maxburnsteps=10000,\
    observables={'E':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1))+E_kin_local(X))},\
    trueenergies=[ef.totalenergy(5)],\
    burn_avg_of=1000)

#batch.name2='total energy for true ground state'
#batch.task2=estimateobservables.Run
#batch.genprofile1=lambda _: estimateobservables.getdefaultprofile().butwith(\
#    wavefunction=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval,\
#    observables={'E':lambda X:jnp.sum(X**2,axis=(-2,-1))+energy.genlocalkinetic()}
#    trueenergy=ef.totalenergy(5))
#batch.task2=harmonicoscillator1d.Run
#batch.genprofile2=lambda prevoutputs: harmonicoscillator1d.getdefaultprofile()



if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)
