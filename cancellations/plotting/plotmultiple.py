import matplotlib.pyplot as plt
import os
from pathlib import Path

from cancellations.display import _display_
from cancellations.utilities import sysutil, tracking, batchjob, browse, numutil, setup
from cancellations.plotting import traingraphs


def trainplots(process,profile):

    plotoptions=profile.plotoptions
    nplots=len(plotoptions)

    runpaths=process.runpaths
    
    fig,axs=plt.subplots(nplots,1,figsize=(8,5*nplots))
    #axs=deque(axs)
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

    #outpath=os.path.join('outputs/combined',' | '.join(process.descriptions))
    #sysutil.savefig(os.path.join(outpath,'comparison.pdf'),fig=fig)

    outpath=os.path.join('outputs/combined',' | '.join(process.descriptions)+'.pdf')
    sysutil.savefig(outpath,fig=fig)
    sysutil.showfile(outpath)

class Run(batchjob.Batchjob):
    processname='plot multiple'
    def runbatch(self):

        P=tracking.Profile()

        pathprofile=tracking.Profile(\
            parentfolder='./',\
            regex='.*outputs.*',\
            condition=lambda path:os.path.exists(path+'/data/setup'))

        allrunpaths=browse.getpaths(pathprofile)
        allprofilepaths=list(set([os.path.join(*Path(p).parts[:-1]) for p in allrunpaths]))

        browsingprocess=browse.Browse(browse.Browse.getdefaultprofile().butwith(\
            onlyone=False,\
            msg='Please select plots to make (with SPACE)'+browse.msg2,\
            options=allprofilepaths,\
            displayoption=lambda path: Path(path).parts[-1],\
            readinfo=lambda path: path.replace('/','\n'),\
            ))
        profilepaths=self.run_subprocess(browsingprocess,taskname='choose run')
        self.descriptions=[Path(path).parts[-1] for path in profilepaths]

        self.runpaths=[]
        for path,desc in zip(profilepaths,self.descriptions):
            runpaths=[p for p in allrunpaths if Path(path).parts[-1] in Path(p).parts]
            browsingprocess=browse.Browse(browse.Browse.getdefaultprofile().butwith(\
                options=runpaths,\
                msg='Please select run for profile:\n{}'.format(desc)+browse.msg,\
                displayoption=lambda path: os.path.join(*Path(path).parts[-2:]),\
                readinfo=lambda path: sysutil.readtextfile(os.path.join(path,'log.txt')),\
                ))
            self.runpaths.append(self.run_subprocess(browsingprocess,taskname='choose run'))

        options=['loss','|f|','|Af|','|f|/|Af|','|weights|']
        browsingprocess2=browse.Browse(browse.Browse.getdefaultfilebrowsingprofile().butwith(\
            onlyone=False,options=options,msg='select plots to make.'+browse.msg))
        P.plotoptions=self.run_subprocess(browsingprocess2,taskname='choose plot options')

        _display_.leavedisplay(self,lambda: trainplots(self,P))




    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile(**kw).butwith(tasks=['choose run','plot'])



if __name__=='__main__':
    Run().run_as_main()
    setup.run_afterdisplayclosed()