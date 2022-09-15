from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d, estimateobservables
from cancellations.examples import estimateobservables_precomputedsamples as ep
from cancellations.functions import examplefunctions as ef
from cancellations.display import cdisplay
from cancellations.utilities import sampling, tracking, arrayutil, browse, batchjob, energy, sysutil
import jax
import jax.numpy as jnp
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')


batch.name1='pick run'
batch.task1=browse.Browse
batch.genprofile1=lambda _: browse.getdefaultprofile().butwith(onlyone=True)




batch.name2='extract learner'
batch.task2=tracking.newprocess(lambda process: sysutil.load(process.path).learner.restore())
batch.genprofile2=lambda prevoutputs: tracking.Profile(path=prevoutputs[-1]+'data/unprocessed')




batch.name3='pick samples from true ground state'
batch.task3=browse.Browse
batch.genprofile3=lambda _: browse.getdefaultprofile().butwith(onlyone=True,
    condition1=lambda path:\
        any(['Total energy K+V for true ground state' in l for l in sysutil.read(path+'metadata.txt')]),\
        readinfo=lambda path: '\n'.join(os.listdir(path)))




batch.name4='observables on learned psi~, loaded samples'
batch.task4=estimateobservables.Run


def genprofile(prevoutputs):

    psi_=prevoutputs[1].restore().f
    E_kin_local=arrayutil.fixparams(energy.genlocalkinetic(psi_),prevoutputs[1].weights)

    psi=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval
    psi_=prevoutputs[1].eval

    sampler=sampling.LoadedSamplesPipe(prevoutputs[-1])

    return estimateobservables.getdefaultprofile().butwith(\
        qpratio=jax.jit(lambda X: psi_(X)**2/psi(X)**2),\
        observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1))),'K':E_kin_local},\
        trueenergies=2*[ef.totalenergy(5)/2],\
        sampler=sampler
        )

batch.genprofile4=genprofile





#
if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)

#    outputs=cdisplay.session_in_display(batchjob.Batchjob,batch)
#    psi_=outputs[1].restore().f
#    E_kin_local=arrayutil.fixparams(energy.genlocalkinetic(psi_),outputs[1].weights)
#
#    psi=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval
#    psi_=outputs[1].eval
#    qpratio=jax.jit(lambda X: psi_(X)**2/psi(X)**2)
#
#    observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1))),'K':E_kin_local}
#    profile=tracking.Profile(qpratio=qpratio,Xs=outputs[3],observables=observables)
#    energies=ep.execprocess(profile)
#    print(energies)

