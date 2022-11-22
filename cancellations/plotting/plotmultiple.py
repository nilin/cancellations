import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as rnd

from cancellations.display import _display_
from cancellations.utilities import textutil
from ..utilities import sysutil, tracking, batchjob, browse, numutil, setup
from ..functions import examplefunctions3d
from . import traingraphs
import random
from collections import deque

def trainplots(process,profile):

    plotoptions=profile.plotoptions
    nplots=len(plotoptions)

    runpaths=process.runpaths
    outpath='outputs/combined/'+str(random.randint(10**9,10**10))+'/'
    
    fig,axs=plt.subplots(nplots,1,figsize=(8,10))
    #axs=deque(axs)
    colors=['r:','b']*10

    for ax,po in zip(axs,plotoptions):
        ax.set_title(po)
        ax.set_yscale('log')
        ax.grid(True,which='major',axis='y')

    for c,runpath,desc in zip(colors,runpaths,process.descriptions):
        i_s,losses,weightnorms,(Afs,fs)=traingraphs.graphdata(process,runpath)

        for ax,po in zip(axs,plotoptions):
            match po:
                case 'loss':
                    ax.plot(i_s,losses,c,label=desc)
                case '|Af|':
                    ax.plot(i_s,Afs,c,label=desc)
                case '|f|':
                    ax.plot(i_s,fs,c,label=desc)
                case '|f|/|Af|':
                    ax.plot(i_s,[f/Af for f,Af in zip(fs,Afs)],c,label=desc)
                case '|weights|':
                    ax.plot(i_s,weightnorms,c,label=desc)

            ax.legend()

    sysutil.savefig(outpath+'comparison.pdf',fig=fig)
    sysutil.showfile(outpath)

class Run(batchjob.Batchjob):
    processname='plot multiple'
    def runbatch(self):

        P=tracking.Profile()

        browsingprocess=browse.Browse(browse.Browse.getdefaultfilebrowsingprofile().butwith(onlyone=False))
        relrunpaths=self.run_subprocess(browsingprocess,taskname='choose run')
        self.runpaths=['outputs/'+relrunpath for relrunpath in relrunpaths]
        self.outpath='outputs/'+' and '.join(relrunpaths).replace('/','_')
        self.descriptions=[a[-50:] for a in relrunpaths]

        options=['loss','|f|','|Af|','|f|/|Af|','|weights|']
        browsingprocess2=browse.Browse(browse.Browse.getdefaultfilebrowsingprofile().butwith(onlyone=False,options=options))
        P.plotoptions=self.run_subprocess(browsingprocess2,taskname='choose plot options')

        _display_.leavedisplay(self,lambda: trainplots(self,P))




    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile(**kw).butwith(tasks=['choose run','plot'])



if __name__=='__main__':
    Run().run_as_main()
    setup.run_afterdisplayclosed()