import jax.numpy as jnp
import jax.random as rnd
from ..utilities import numutil as mathutil, tracking
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')




def samplepoints(X,Y,nsamples):
    p=Y**2
    p=p/jnp.sum(p)
    I=rnd.choice(tracking.currentprocess().nextkey(),jnp.arange(len(p)),(nsamples,),p=p)
    return X[I]
    

def linethrough(x,interval):
    corner=np.zeros_like(x)
    corner[0][0]=1
    x_rest=(1-corner)*x
    X=interval[:,None,None]*corner[None,:,:]+x_rest[None,:,:]
    return X







#def genCrossSections(X,Y,target):

class CrossSection:
    info=''
    #def __init__(self,X,Y,interval):
    def __init__(self,X,Y,interval=jnp.arange(-1,1,2/100)):
        self.interval=interval
        self.X=X
        self.Y=Y
        self.n=X.shape[-2]
    def plot(self,staticlearner,**kwargs):
        f=mathutil.closest_multiple(staticlearner,self.X[:1000],self.Y[:1000])
        return self.plot_y_vs_f(f,**kwargs)


class Line_1particle(CrossSection):
    def __init__(self,X,Y,target,x0,**kw):
        super().__init__(X,Y,**kw)
        self.line=linethrough(x0,self.interval)
        self.y=target(self.line)

    def plot(self,f,normalized_target=False):

        c=1/mathutil.norm(self.Y) if normalized_target else 1

        fig,ax=plt.subplots()
        ax.plot(self.interval,c*self.y,'b',label='target')
        ax.plot(self.interval,f(self.line),'r',ls='dashed',label='learned')
        ax.legend()
        return fig

    
class Slice(CrossSection):

    def __init__(self,X,Y,target,x0,**kw):
        super().__init__(X,Y,**kw)
        self.slice=self.slicethrough(x0,self.interval)
        self.y=mathutil.applyalonglast(target,self.slice,2)


    def plot(self,f):
        I=self.interval
        #c=1/mathutil.norm(self.Y)

        fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(15,6))
        #yt=c*self.y
        yt=self.y
        yl=mathutil.applyalonglast(f,self.slice,2)

        #M=1 if normalized_target else mathutil.norm(self.Y)
        #M*=4

        ax0.set_title('target')
        ax1.set_title('learner')
        ax2.set_title('both')

        levels=[-.1,.1]

        mt=ax0.pcolormesh(I,I,yt,cmap='seismic')#,vmin=-M,vmax=M)
        ct=ax0.contour(I,I,yt,levels=levels,colors='k',linewidths=1)
        #cl=ax0.contour(I,I,yl,levels=levels,colors='k',linewidths=.1,alpha=.5)
        #plt.clabel(cl,inline=True,fmt=lambda x:'learner')

        ml=ax1.pcolormesh(I,I,yl,cmap='seismic')#,vmin=-M,vmax=M)
        cl=ax1.contour(I,I,yl,levels=levels,colors='k',linewidths=1)
        #ct=ax1.contour(I,I,yt,levels=levels,colors='k',linewidths=.1,alpha=.5)
        #plt.clabel(ct,inline=True,fmt=lambda x:'target')

        for m in [mt,ml]: m.set_edgecolor('face')

        c0=ax2.contour(I,I,yt,levels=levels,colors='b',linewidths=2)
        c1=ax2.contour(I,I,yl,levels=levels,colors='r',linewidths=1)
        plt.clabel(c0,inline=True,fmt=lambda x:'target')

        for ax in (ax0,ax1,ax2): ax.set_aspect('equal')

        return fig



class Slice_1particle(Slice):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.info='one particle free, {} particles fixed'.format(self.n-1)

    @staticmethod
    def slicethrough(x,I):
        S,T=np.meshgrid(I,I)
        X=np.array(x)[None,None,:,:]+np.zeros_like(S)[:,:,None,None]
        X[:,:,0,0]=S
        X[:,:,0,1]=T
        return X


class Slice_2particles(Slice):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.info='two particles free, {} particles fixed'.format(self.n-2)

    @staticmethod
    def slicethrough(x,I):
        S,T=np.meshgrid(I,I)
        X=np.array(x)[None,None,:,:]+np.zeros_like(S)[:,:,None,None]
        X[:,:,0,0]=S
        X[:,:,1,0]=T
        return X



############

def genCrossSections(targetfn,**kw):
    #tracking.logcurrenttask('Preparing cross sections for plotting.')    
    tracking.currentkeychain=4

    X=tracking.currentprocess().genX(1000)
    Y=targetfn(X)
    n=X.shape[-1]
    x0s=samplepoints(X,Y,{1:3,2:3,3:1}[n])
    sections1p=[Slice_1particle(X,Y,targetfn,x0,**kw) for x0 in x0s] if n>1 else []
    sections2p=[Slice_2particles(X,Y,targetfn,x0,**kw) for x0 in x0s]
    #tracking.clearcurrenttask()
    return *sections2p,*sections1p

#