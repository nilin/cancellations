import jax.numpy as jnp
import jax.random as rnd
import os
import copy
from os import path
import numpy as np
from functools import partial, reduce
from cancellations.config import config as cfg, sysutil, tracking, browse
from cancellations.config.browse import Browse
from cancellations.functions import _functions_
from cancellations.functions._functions_ import ComposedFunction,SingleparticleNN,Product
from cancellations.run import sampling
from cancellations.utilities import numutil, textutil
import matplotlib.pyplot as plt
from cancellations.config.tracking import dotdict, log
from cancellations.display import _display_
from cancellations.examples import losses
from jax.tree_util import tree_map
import optax


class Run(_display_.Process):

    processname='example template'
    processtype='runs'

    def execprocess(run):
        P=run.profile
        run.info='runID: {}\n'.format(run.ID)+'\n'*4+run.parseinfo(P.info)
        if cfg.display_on: run.prepdisplay()
        for path in [P.outpath_data,P.outpath_plot]:
            sysutil.write(run.info,os.path.join(path,'info.txt'),mode='w')
            sysutil.write(P.name,os.path.join(path,'profilename.txt'),mode='w')

        run.losses={ln:[] for ln in P.lossnames}
        if cfg.display_on: run.addlearningdisplay()

        stopwatch1=tracking.Stopwatch()

        opt=optax.adamw(learning_rate=.01,weight_decay=P.weight_decay)
        state=opt.init(P.learner.weights)
        run.its=0

        run.prep(P)
        log('start training')

        for i in range(P.iterations+1):
            try:
                run.its=i
                X,*Ys=P.sampler.sample()

                Ls,Gs=zip(*[lossgrad(P.learner.weights,X,*Ys) for lossgrad in P.lossgrads])
                for ln,loss in zip(P.lossnames,Ls):
                    run.losses[ln].append(loss)
                grad=sumgrads(rescale(P.lossweights,Gs))

                updates,state=opt.update(grad,state,P.learner.weights)
                P.learner.weights=optax.apply_updates(P.learner.weights,updates)

                run.repeat(i)

                if stopwatch1.tick_after(.05):
                    if cfg.display_on: run.learningdisplay.draw()
                    if run.act_on_input(tracking.getch(run.getinstructions))=='b': break

            except KeyboardInterrupt:
                print()
                print(run.getinstructions())
                print()
                run.act_on_input(input())

        return run.finish(P)

    def act_on_input(self,key):
        if key=='q': quit()
        if key=='p': self.plot(self.profile)
        if key=='d': breakpoint()
        if key=='o':
            sysutil.showfile(self.profile.outpath_data)
            sysutil.showfile(self.profile.outpath_plot)
        return key

    def subprocess(self,process):
        out=tracking.runprocess(process)
        try: self.T.draw()
        except: pass
        return out

    def plot(self,P,currentrun=True,parsefigname=lambda a:'{} loss'.format(a),parselinename=lambda a:'{} profile'.format(a)):
        plotoptions=self.subprocess(Browse(options=self.lossnames,onlyone=False,msg='Select statistics to be plotted'))
        nplots=len(plotoptions)
        fig,axs=plt.subplots(nplots,1,figsize=(8,5*nplots))
        if nplots==1: axs=[axs]

        if currentrun:
            losses_=[self.losses]
            profiles=[P.name]
        else:
            paths,pathstrings,allprofiles=[],[],[]
            for relpath in sorted(os.listdir('outputs')):
                try:
                    path=os.path.join('outputs',relpath)
                    pname=open(os.path.join(path,'profilename.txt')).readline()
                    pathstring=path+' '+pname
                except:
                    continue
                paths.append(path)
                pathstrings.append(pathstring)
                allprofiles.append(pname)

            runs=self.subprocess(Browse(options=list(zip(paths,allprofiles)),onlyone=False,optionstrings=pathstrings,msg='Select runs'))
            runs,profiles=zip(*runs)
            losses_=[sysutil.load(os.path.join(path,'losses')) for path in runs]

        fig.suptitle(' vs '.join([parselinename(p) for p in profiles]))

        colors=['b','r','b--','r--']
        colors_=[self.subprocess(Browse(options=colors,msg='pick color for {}'.format(I))) for I in profiles]
        smoothing=self.subprocess(Browse(options=[1,10,100],msg='pick smoothing'))
        T0=max([len(l[po]) for l in losses_ for po in plotoptions])
        T1=min([len(l[po]) for l in losses_ for po in plotoptions])
        T=self.subprocess(Browse(msg='pick duration',options=\
            sorted([T0,T1]+[a*x for x in [1000,10000,100000] for a in range(2,20) if a*x<T1])))

        for ax,po in zip(axs,plotoptions):
            ax.set_title(parsefigname(po))
            ax.set_yscale('log')
            ax.grid(True,which='major',axis='y')
            for i,(I_f,c) in enumerate(zip(profiles,colors_)):
                smoother=numutil.RunningAvg(k=smoothing)
                ax.plot([smoother.update(l) for l in losses_[i][po][:T]],c,label=parselinename(I_f))
                ax.legend()

        outpath=os.path.join('plots','training_comp_{}.pdf'.format(tracking.nowstr()))
        sysutil.savefig(outpath,fig=fig)
        sysutil.showfile(outpath)

#    def plot(self,P):
#        options=['heatmaps','contours']
#        plotoptions=tracking.runprocess(browse.Browse(onlyone=False,options=options))
#        return
#        if 'heatmaps' in plotoptions or 'contours' in plotoptions:
#            I,Xp,Yp=P.plotslice
#            M=jnp.quantile(jnp.abs(Yp),.9)
#            fp=P.learner.eval(Xp)
#            fp=fp*jnp.dot(fp,Yp)/jnp.dot(fp,fp)
#            P.plotslice_=(I,Xp,Yp,fp)
#        for o in ['heatmaps','contours']:
#            match o:
#                case 'heatmaps':
#                    I,Xp,Yp,fp=P.plotslice_
#                    M=jnp.quantile(jnp.abs(Yp),.9)
#                    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(15,6))
#                    handle0=ax0.pcolormesh(I,I,Yp,cmap='seismic',vmin=-M,vmax=M)
#                    handle1=ax1.pcolormesh(I,I,fp,cmap='seismic',vmin=-M,vmax=M)
#                    for m in [handle0,handle1]: m.set_edgecolor('face')
#                    ct0=ax0.contour(I,I,Yp,levels=[-10,-1,-.1,0,.1,1,10],colors='k',linewidths=1)
#                    ct1=ax1.contour(I,I,fp,levels=[-10,-1,-.1,0,.1,1,10],colors='k',linewidths=1)
#                    for ax in [ax0,ax1]: ax.set_aspect('equal')
#                case 'contours':
#                    I,Xp,Yp,fp=P.plotslice_
#                    fig,(ax)=plt.subplots(1,1,figsize=(6,6))
#                    ct0=ax.contour(I,I,Yp,levels=[-10,-1,-.1,0,.1,1,10],colors='b',linewidths=2)
#                    ct1=ax.contour(I,I,fp,levels=[-10,-1,-.1,0,.1,1,10],colors='r',linewidths=1)
#                    ax.set_aspect('equal')
#
#            outpath=os.path.join(P.outpath_plot,'{}.pdf'.format(o))
#            sysutil.savefig(outpath)
#            sysutil.showfile(outpath)

    def repeat(run,i):
        if i is None: return
        if i%100==0:
            log('{} iterations'.format(i))
        if i%25==0:
            log(' '.join(['{}={:.3f}'.format(k,v[-1]) for k,v in run.losses.items()]))
        if i%1000==0:
            sysutil.save(run.losses,os.path.join(run.profile.outpath_data,'losses'))
        if not cfg.display_on:
            if i%1000==0:
                print('\nnodisplay mode--raise KeyboardInterrupt (typically Ctrl-C) to pause and view options\n')

    @staticmethod
    def prep(P):
        pass

    @staticmethod
    def finish(P):
        pass

    @staticmethod
    def getinstructions():
        return 'Press:'+\
            '\n[p] to generate plots.'+\
            '\n[o] to open output folder.'+\
            '\n[b] to break from current task.'+\
            '\n[d] to enter pdb'+\
            '\n[q] to quit.'

    def prepdisplay(self):
        instructions=self.getinstructions()

        self.T,self.learningdisplay=self.dashboard.vsplit(rlimits=[.8])
        self.L,self.R=self.T.hsplit()

        self.L.add(0,0,_display_._TextDisplay_(instructions))
        self.L.add(0,20,_display_._LogDisplay_(self.L.width,25,balign=False))
        self.L.vstack()

        self.infodisplay=self.R.add(0,0,_display_._TextDisplay_(self.info))

        self.T.arm()
        self.learningdisplay.arm()
        self.T.draw()

    def addlearningdisplay(self):
        k=10
        lossnames=self.losses.keys()
        smoothers={ln:numutil.RunningAvg(k=k) for ln in lossnames}
        display=self.learningdisplay.add(0,0,_display_._Display_())
        def bar_at(lossname,smoother,loss,k,pos):
            return [(0,3*pos+3,'{} (smoothed over {} iterations) {:.2E}'.format(lossname,k,smoother.update(loss))),\
            (0,3*pos+4,_display_.hiresbar(smoother.avg(),self.dashboard.width))]
            
        encode=lambda:[bar_at(ln,smoothers[ln],self.losses[ln][-1],k,pos=i) for i,ln in enumerate(lossnames)]
        display.encode=lambda: [(0,0,'{:,d}/{:,d} iterations'.format(self.its,self.profile.iterations))]\
                                +[s for l in encode() for s in l]+[(0,1,'')]

    @staticmethod
    def getslice(X,Y,rho):
        x=X[jnp.argmax(rho*Y**2)]
        I=jnp.arange(-3,3.03,.03)
        return slicethrough(x,I),I

    @staticmethod
    def parseinfo(I):
        return '\n'.join(['{}:{}'.format(k,v) for k,v in I.items()])

    @classmethod
    def getprofiles(cls):
        return {'default':cls.getdefaultprofile}

    @staticmethod
    def defaultbaseprofile():
        P=tracking.Profile()

        P.outpath_data=os.path.join('outputs',tracking.sessionID)
        P.outpath_plot=os.path.join('outputs',tracking.sessionID)

        P.weight_decay=0.0
        P.iterations=10**5
        P.samples_train=10**6
        P.samples_test=1000
        P.evalblocksize=10**4

        P.info=dict()
        P.name='base profile'
        return P


class Fixed_X(Run):
    @classmethod
    def getprofile(cls,n,d,samples_train,minibatchsize):
        P=cls.defaultbaseprofile()
        cls.prep_X(P,n,d,samples_train,minibatchsize)
        return P.butwith(n=n,d=d,samples_train=samples_train,minibatchsize=minibatchsize)

    @staticmethod
    def prep_X(P,n,d,samples_train,minibatchsize):
        P._var_X_distr_=1
        P._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(P._var_X_distr_)
        P.X_density=numutil.gen_nd_gaussian_density(var=P._var_X_distr_)
        P.X=P._genX_(rnd.PRNGKey(0),samples_train,n,d)
        P.rho=P.X_density(P.X)
        
    @staticmethod
    def initsampler(P):
        P.sampler=sampling.SamplesPipe(P.X,P.Y,P.rho,minibatchsize=P.minibatchsize)

class Fixed_XY(Fixed_X):
    @classmethod
    def getprofile(cls,n,d,samples_train,minibatchsize,target):
        P=super().getprofile(n,d,samples_train,minibatchsize)
        P.Y=numutil.blockwise_eval(target,blocksize=P.evalblocksize,msg='preparing training data')(P.X)
        return P

class Loaded_XY(Run):
    @classmethod
    def getprofile(cls,X,Y,rho,minibatchsize):
        samples_train,n,d=X.shape
        P=cls.defaultbaseprofile().butwith(\
            n=n,d=d,samples_train=samples_train,minibatchsize=minibatchsize,\
            X=X,Y=Y,rho=rho)
        return P

def slicethrough(x,I):
    S,T=np.meshgrid(I,I)
    X=np.array(x)[None,None,:,:]+np.zeros_like(S)[:,:,None,None]
    X[:,:,0,0]=S
    X[:,:,0,1]=T
    return X

def rescale(cs,Gs):
    return [tree_map(lambda A:c*A,G) for c,G in zip(cs,Gs)]

def sumgrads(Gs):
    add=lambda *As: reduce(jnp.add,As)
    return tree_map(add,*Gs)
    
