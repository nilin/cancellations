import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as rnd

from cancellations.display import _display_
from cancellations.utilities import textutil
from ..utilities import sysutil, tracking, batchjob, browse, numutil, setup
from ..functions import examplefunctions3d
from . import traingraphs



class Slice:

    def plot(self,ax,f_eval,f_descr):
        Y=jax.vmap(f_eval)(self.X)
        m=ax.pcolormesh(Y,cmap='seismic')
        m.set_edgecolor('face')
        ax.set_title(f_descr.richtypename())
        ax.set_aspect('equal')

    def contour(self,ax,f_eval,*args,**kw):
        Y=jax.vmap(f_eval)(self.X)
        ax.contour(Y,*args,levels=[-1,0,1],**kw)
        ax.set_aspect('equal')

    def compare(self,*fs):

        xnorm=rnd.normal(tracking.nextkey(),(100,self.n,self.d)) 
        f0=numutil.normalize(fs[0].eval,xnorm)
        evals=[numutil.closest_multiple(f.eval,xnorm,f0(xnorm)) for f in fs]


        fig,axs=plt.subplots(1,len(fs),figsize=(7*len(fs),7))
        for ax,f_eval,f_descr in zip(axs,evals,fs):
            self.plot(ax,f_eval,f_descr)
            self.contour(ax,f_eval,colors='k')
        
        sysutil.savefig(self.process.outpath+'{}_heatmaps.pdf'.format(self.slicetype),fig=fig)

        fig,ax=plt.subplots(figsize=(7,7))
        for f_eval,c,lw in zip(evals,textutil.colors,[2,1,.5,.25]):
            self.contour(ax,f_eval,colors=c,linewidths=lw)

        sysutil.savefig(self.process.outpath+'{}_contours.pdf'.format(self.slicetype),fig=fig)


class Slice_1p(Slice):
    slicetype='1particle'

    def __init__(self,process,surface,x2__):
        self.process=process

        w,h,self.d=surface.shape
        self.n=x2__.shape[0]+1
        X2__=x2__[None,None,:,:]+jnp.zeros((w,h))[:,:,None,None]

        self.X=jnp.stack([surface]+[X2__[:,:,i,:] for i in range(self.n-1)],axis=-2)


class Slice_2p(Slice):
    slicetype='2particle'

    def __init__(self,process,curve,x3__):
        self.process=process

        l,self.d=curve.shape
        self.n=x3__.shape[0]+2
        x1=curve[:,None,:]+jnp.zeros_like(curve)[None,:,:]
        x2=curve[None,:,:]+jnp.zeros_like(curve)[:,None,:]
        X3__=x3__[None,None,:,:]+jnp.zeros((l,l))[:,:,None,None]

        self.X=jnp.stack([x1,x2]+[X3__[:,:,i,:] for i in range(self.n-2)],axis=-2)





class RandomSlices:

    def __init__(self,process,n,d,r=5,fineness=100):
        self.process=process

        x0=rnd.normal(tracking.nextkey(),(n,d))
        I=jnp.arange(-r,r,2*r/fineness)
        l=len(I)
        diagonalcurve=I[:,None]+jnp.zeros((d,))[None,:]

        X1,X2=jnp.meshgrid(I,I)
        plane=jnp.stack([X1,X2]+[jnp.zeros((l,l)) for i in range(d-2)],axis=-1)

        self.s1=Slice_1p(process,plane,x0[1:])
        self.s2=Slice_2p(process,diagonalcurve,x0[2:])

    def plot(self,*fs):
        for s in [self.s1,self.s2]:
            s.compare(*fs)


def allplots(process):
    runpath=process.outpath

    target=sysutil.load(runpath+'data/target').restore()
    learner=sysutil.load(runpath+'data/learner').restore()
    setupdata=sysutil.load(runpath+'data/setup')
    _,n,d=setupdata['X_test'].shape

    process.log('generating heatmaps and contour plots')

    S=RandomSlices(process,n,d)
    S.plot(target,learner)
    sysutil.showfile(process.outpath)

    traingraphs.graph(process,runpath)


class Run(batchjob.Batchjob):
    processname='plotting'
    def runbatch(self):

        browsingprocess=browse.Browse(browse.Browse.getdefaultfilebrowsingprofile())
        runpath='outputs/'+self.run_subprocess(browsingprocess,taskname='choose run')
        self.outpath=runpath

        process,display=self.loadprocess(taskname='plot')
        info=sysutil.readtextfile(runpath+'info.txt')
        display.add(0,0,_display_._TextDisplay_(info))
        display.arm()
        display.draw()
        #sysutil.write(info,process.outpath+'info.txt')

        allplots(self)

        #def postrun():
        #    tracking.loadprocess(tracking.Process())
        #    traingraphs.graph(self,runpath)
        #setup.postrun=postrun 
        #return 




    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile(**kw).butwith(tasks=['choose run','plot'])



