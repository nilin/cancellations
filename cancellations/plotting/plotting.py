import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as rnd
from cancellations.config import batchjob, browse, config as cfg, sysutil, tracking
from os import path

from cancellations.display import _display_
from cancellations.utilities import textutil
from cancellations.utilities import numutil
from cancellations.plotting import traingraphs


# dontpick

class Slice:

    def plot(self,ax,f_eval,f_descr):
        Y=jax.vmap(f_eval)(self.X)
        m=ax.pcolormesh(Y,cmap='seismic')
        m.set_edgecolor('face')
        ax.set_title(f_descr.richtypename())
        ax.set_aspect('equal')

    def contour(self,ax,f_eval,*args,levels=[0],**kw):
        Y=jax.vmap(f_eval)(self.X)
        ax.contour(Y,*args,levels=levels,**kw)
        ax.set_aspect('equal')

    def compare(self,*fs):

        xnorm=rnd.normal(rnd.PRNGKey(0),(100,self.n,self.d)) 
        f0=numutil.normalize(fs[0].eval,xnorm)
        evals=[numutil.closest_multiple(f.eval,xnorm,f0(xnorm)) for f in fs]


        fig,axs=plt.subplots(1,len(fs)+1,figsize=(8*(len(fs)+1),8))
        for ax,f_eval,f_descr,c,lw in zip(axs[:-1],evals,fs,textutil.colors,[3**(-k) for k in range(10)]):
            self.plot(ax,f_eval,f_descr)
            self.contour(ax,f_eval,colors='k')

            self.contour(axs[-1],f_eval,colors=c,linewidths=5*lw)
            levels=[2**(k/2) for k in range(-10,10)]
            levels=list(reversed([-l for l in levels]))+levels
            self.contour(axs[-1],f_eval,colors=c,levels=levels,linewidths=lw,alpha=.5)
        
        sysutil.savefig(path.join(self.process.outpath,'{}.pdf'.format(self.slicetype)),fig=fig)



class Slice_1p(Slice):
    slicetype='1particle'

    def __init__(self,process,surface,x2__):

        process.log('generating 1-particle slice')
        self.process=process

        w,h,self.d=surface.shape
        self.n=x2__.shape[0]+1
        X2__=x2__[None,None,:,:]+jnp.zeros((w,h))[:,:,None,None]

        self.X=jnp.stack([surface]+[X2__[:,:,i,:] for i in range(self.n-1)],axis=-2)


class Slice_2p(Slice):
    slicetype='2particle'

    def __init__(self,process,curve,x3__):

        process.log('generating 2-particle slice')
        self.process=process

        l,self.d=curve.shape
        self.n=x3__.shape[0]+2
        x1=curve[:,None,:]+jnp.zeros_like(curve)[None,:,:]
        x2=curve[None,:,:]+jnp.zeros_like(curve)[:,None,:]
        X3__=x3__[None,None,:,:]+jnp.zeros((l,l))[:,:,None,None]

        self.X=jnp.stack([x1,x2]+[X3__[:,:,i,:] for i in range(self.n-2)],axis=-2)





class RandomSlices:

    def __init__(self,process,n,d,r=3,fineness=100):
        self.process=process

        x0=rnd.normal(rnd.PRNGKey(1),(n,d))
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

    target=sysutil.load(path.join(runpath,'data/target')).restore()
    learner=sysutil.load(path.join(runpath,'data/learner')).restore()
    setupdata=sysutil.load(path.join(runpath,'data/setup'))
    _,n,d=setupdata['X_test'].shape

    process.log('generating heatmaps and contour plots')

    S=RandomSlices(process,n,d)
    S.plot(target,learner)

    traingraphs.graph(process,runpath)
    sysutil.showfile(process.outpath)


class Run(batchjob.Batchjob):
    processname='plotting'
    def execprocess(self):

        browsingprocess=browse.Browse(browse.Browse.getdefaultprofile())
        relrunpath=self.run_subprocess(browsingprocess,taskname='choose run')

        if relrunpath is None:
            cfg.postcommand=lambda: print('No outputs found')
            return

        runpath=path.join('outputs/',relrunpath)
        self.outpath=runpath

        _display_.leavedisplay(self,lambda: allplots(self))




    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile(**kw).butwith(tasks=['choose run','plot'])



