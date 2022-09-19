from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d, estimateobservables, profiles
from cancellations.testing import testing
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay
from cancellations.utilities import numutil, sampling, tracking, browse, batchjob, energy, sysutil
from cancellations.utilities.sysutil import maybe as maybe
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os




profile=tracking.Profile(tasks=[\
    'pick training run',\
    'estimate observable',\
    ])


class Run(batchjob.Batchjob):
    def execprocess(batch):

        # task 1

        pathprofile=browse.defaultpathprofile()
        fullpaths=['outputs/'+relpath for relpath in browse.getpaths(pathprofile)]
        fullpaths=sorted(fullpaths,key=lambda full: -os.path.getmtime(full))


        bprofile=browse.Browse.getdefaultprofile().butwith(\
                onlyone=True,\
                options=fullpaths,\
                readinfo=lambda full: sysutil.readtextfile(full+'/info.txt'),\
                displayoption=lambda full: full[8:]+' '+maybe(sysutil.readtextfile,'')(full+'/metadata.txt') )

        runpath=batch.runsubprocess(browse.Browse(**bprofile),name='pick training run')


        # task 2

        psi_descr=sysutil.load(runpath+'data/setup')['learner'].restore()
        X=unprocessed=sysutil.load(runpath+'data/setup')['X_train'][:1000]
        unprocessed=sysutil.load(runpath+'data/unprocessed')
        _psi_=psi_descr._eval_
        fdescr,switchcounts=functions.switchtype(psi_descr)
        _f_=fdescr._eval_

        assert(switchcounts==1)
        testing.verify_antisymmetrization(psi_descr.eval,fdescr.eval,X[:100])

        # change this to be from saved data
        _X_distr_density_=profiles.getprofiles('harmonicoscillator1d')['default']()._X_distr_density_

        weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')

        def getnorm(_f_,weights,X):
            Y=_f_(weights,X)
            squaresums=jnp.sum(Y**2,axis=Y.shape[1:])
            normalized=squaresums/_X_distr_density_(X)
            return jnp.average(normalized)

        Afs=[getnorm(_psi_,weights,X) for weights in weightslist]
        fs=[getnorm(_f_,weights,X) for weights in weightslist]
        f_over_Af=[f/Af for f,Af in zip(fs,Afs)]

        fig1,ax=plt.subplots()
        ax.plot(i_s,f_over_Af)
        ax.set_yscale('log')
        fig1.suptitle('|f|^2/|Af|^2'+sysutil.maybe(lambda:'\nprofile name: '+sysutil.load(runpath+'data/setup')['profilename'],'')())

        fpath=batch.outpath+'ratio.pdf' 
        sysutil.savefig(fpath,fig=fig1)
        sysutil.showfile(batch.outpath)
        sysutil.showfile(fpath)

        #plt.show()

        info=sysutil.readtextfile(runpath+'info.txt')
        sysutil.write(info,batch.outpath+'info.txt')

Run(**profile).run_as_main()
#