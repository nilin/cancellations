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