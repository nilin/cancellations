import matplotlib.pyplot as plt
import os
from pathlib import Path
from cancellations.config import browse, config as cfg, tracking

from cancellations.display import _display_
from cancellations.config import sysutil
from cancellations.plotting import traingraphs



def trainplots(process,profile):

    plotoptions=profile.plotoptions
    nplots=len(plotoptions)

    runpaths=process.runpaths

    fig,axs=plt.subplots(nplots,1,figsize=(8,5*nplots))
    if nplots==1: axs=[axs]
    colors=['r--','b']*10

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

    outpath=os.path.join('outputs/combined',' | '.join(process.descriptions)+'.pdf')
    sysutil.savefig(outpath,fig=fig)
    sysutil.showfile(outpath)

class Run(_display_.Process):
    processname='plot multiple'
    def execprocess(self):

        P=cfg.Profile()


        rdir='outputs/runs'
        runpaths=[os.path.join(rdir,p) for p in os.listdir(rdir)]
        browsingprocess=browse.Browse(browse.Browse.getdefaultprofile().butwith(\
            options=runpaths,\
            msg='Please select run\n'+browse.msg,\
            displayoption=lambda path: os.path.join(*Path(path).parts[-2:]),\
            readinfo=lambda path: sysutil.readtextfile(os.path.join(path,'log.txt')),\
            onlyone=False,\
            ))
        self.runpaths=self.run_subprocess(browsingprocess,taskname='choose run')

        self.descriptions=self.runpaths # replace with profile description

        options=['loss','|f|','|Af|','|f|/|Af|','|weights|']
        browsingprocess2=browse.Browse(browse.Browse.getdefaultprofile().butwith(\
            onlyone=False,options=options,msg='select plots to make.'+browse.msg))
        P.plotoptions=self.run_subprocess(browsingprocess2,taskname='choose plot options')

        _display_.leavedisplay(self,lambda: trainplots(self,P))



if __name__=='__main__':
    Run().run_as_main()
    cfg.run_afterdisplayclosed()