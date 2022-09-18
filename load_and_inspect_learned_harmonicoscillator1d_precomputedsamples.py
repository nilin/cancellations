from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d, estimateobservables
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay
from cancellations.utilities import numutil, sampling, tracking, browse, batchjob, energy, sysutil
from cancellations.utilities.sysutil import maybe as maybe
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

        pathprofile=browse.defaultpathprofile().butwith(regex='.*tgsamples/?',condition1=None)
        relpaths=browse.getpaths(pathprofile)
        fullpaths=['outputs/'+relpath for relpath in relpaths]
        rels={full:rel for full,rel in zip(fullpaths,relpaths)}

        bprofile=browse.Browse.getdefaultprofile().butwith(
            options=fullpaths,\
            onlyone=True,\
            readinfo=lambda path: '\n'.join(sorted(list(os.listdir(path)),key=lambda p:os.path.getmtime(path+p))),\
            displayoption=lambda full: rels[full]+' '+maybe(sysutil.readtextfile,'')(full+'/metadata.txt')
            )
        samplepath=batch.runsubprocess(browse.Browse(**bprofile),name='pick samples')


        # task 2

        pathprofile=browse.defaultpathprofile()
        fullpaths=['outputs/'+relpath for relpath in browse.getpaths(pathprofile)]
        fullpaths=sorted(fullpaths,key=lambda full: os.path.getmtime(full))


        bprofile=browse.Browse.getdefaultprofile().butwith(\
                onlyone=True,\
                options=fullpaths,\
                readinfo=lambda full: sysutil.readtextfile(full+'/info.txt'),\
                displayoption=lambda full: full[8:]+' '+maybe(sysutil.readtextfile,'')(full+'/metadata.txt') )

        runpath=batch.runsubprocess(browse.Browse(**bprofile),name='pick training run')


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