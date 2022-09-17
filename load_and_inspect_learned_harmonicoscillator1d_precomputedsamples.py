from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d, estimateobservables
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay
from cancellations.utilities import numutil, sampling, tracking, browse, batchjob, energy, sysutil
import jax
import jax.numpy as jnp
import os




profile=tracking.Profile(tasks=[\
    'pick samples',\
    'pick training run',\
    'estimate observable',\
    ])


class Run(batchjob.Batchjob):
    def execprocess(batch):

        # task 1

        bprofile=browse.getdefaultprofile().butwith(onlyone=True, regex='.*tgsamples/?',condition1=None,\
            readinfo=lambda path: '\n'.join(sorted(list(os.listdir(path)),key=lambda p:os.path.getmtime(path+p))))
        samplepath=batch.runsubprocess(browse.Browse(**bprofile),name='pick samples')


        # task 2

        runpath=batch.runsubprocess(browse.Browse(**browse.getdefaultprofile().butwith(onlyone=True)),name='pick training run')


        # task 3

        psi_descr=sysutil.load(runpath+'data/unprocessed').learner.restore()

        psi=psi_descr.eval
        E_kin_local=numutil.forfixedparams(energy.genlocalkinetic)(psi)

        q=lambda X:psi(X)**2
        p=sysutil.load(samplepath+'density').restore().eval

        sampler=sampling.LoadedSamplesPipe(samplepath)

        sprofile=estimateobservables.Run.getdefaultprofile().butwith(\
            qpratio=jax.jit(lambda X: jnp.squeeze(q(X))/jnp.squeeze(p(X))),\
            observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1)))},\
            trueenergies={'V':ef.totalenergy(5)/2},\
            #observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1))),'K':E_kin_local},\
            #trueenergies={k:ef.totalenergy(5)/2 for k in ['V','K']},\
            thinningratio=1,\
            #p=p,\
            sampler=sampler,\
            preburnt=True
            )

        batch.runsubprocess(estimateobservables.Run(**sprofile),name='estimate observable')

Run(**profile).run_as_main()
#
#
#
#batch.name1='pick samples from true ground state'
#batch.task1=browse.Browse
#batch.genprofile1=lambda _: 
#
#
#batch.name2='pick run'
#batch.task2=browse.Browse
#batch.genprofile2=lambda _: 
#
#
#
#
#batch.name3='extract learner'
#batch.task3=tracking.newprocess(lambda process: sysutil.load(process.path).learner.restore())
#batch.genprofile3=lambda prevoutputs: tracking.Profile(path=prevoutputs[-1]+)
#
#
#
#
#
#
#
#batch.name4='observables on learned psi~, loaded samples'
#batch.task4=estimateobservables.Run
#
#
#def genprofile(prevoutputs):
#
#batch.genprofile4=genprofile
#
#
#
#
#
##
#if __name__=='__main__':
#    cdisplay.session_in_display(batchjob.Batchjob,batch)
#
##    outputs=cdisplay.session_in_display(batchjob.Batchjob,batch)
##    psi_=outputs[1].restore().f
##    E_kin_local=arrayutil.fixparams(energy.genlocalkinetic(psi_),outputs[1].weights)
##
##    psi=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval
##    psi_=outputs[1].eval
##    qpratio=jax.jit(lambda X: psi_(X)**2/psi(X)**2)
##
##    observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1))),'K':E_kin_local}
##    profile=tracking.Profile(qpratio=qpratio,Xs=outputs[3],observables=observables)
##    energies=ep.execprocess(profile)
##    print(energies)
#
#