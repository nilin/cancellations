import jax.numpy as jnp
import jax.random as rnd
import os
import copy
from os import path
import numpy as np
from functools import partial, reduce
from cancellations.config import config as cfg, sysutil, tracking, browse
from cancellations.functions import _functions_
from cancellations.functions._functions_ import ComposedFunction,SingleparticleNN,Product
from cancellations.run import sampling
from cancellations.utilities import numutil, textutil
import matplotlib.pyplot as plt
from cancellations.config.tracking import dotdict, log
from cancellations.config.batchjob import Batchjob
from cancellations.plotting import plotting
from cancellations.display import _display_
from cancellations.examples import losses
from jax.tree_util import tree_map
import optax


#class Run(_display_.Process):
class Run(Batchjob):

    processname='example template'
    processtype='runs'

    def execprocess(run):
        run.pickprofile()

        run.prepdisplay()
        P=run.profile
        run.info='runID: {}\n'.format(run.ID)+'\n'*4+P.parseinfo(P.info)
        run.infodisplay.msg=run.info
        sysutil.write(run.info,path.join(run.outpath,'info.txt'),mode='w')
        run.T.draw()

        run.traindata=dict()
        run.losses={ln:[] for ln in P.lossnames}
        run.addlearningdisplay()

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        opt=optax.adamw(learning_rate=.01,weight_decay=P.weight_decay)
        state=opt.init(P.learner.weights)
        run.its=0

        P.prep(run)
        log('start training')

        for i in range(P.iterations+1):
            run.its=i
            X,*Ys=P.sampler()

            Ls,Gs=zip(*[lossgrad(P.learner.weights,X,*Ys) for lossgrad in P.lossgrads])
            for i,(ln,loss) in enumerate(zip(P.lossnames,Ls)):
                run.losses[ln].append(loss)
            run.traindata[i]={ln:loss for ln,loss in zip(P.lossnames,Ls)}
            grad=sumgrads(rescale(P.lossweights,Gs))

            updates,state=opt.update(grad,state,P.learner.weights)
            P.learner.weights=optax.apply_updates(P.learner.weights,updates)
            P.repeat(run,i)

            if stopwatch1.tick_after(.05):
                run.learningdisplay.draw()

            if stopwatch2.tick_after(.5):
                if run.act_on_input(cfg.getch(run.getinstructions))=='b': break

        return P.finish(run)

    def getits(run): return run.its

    def act_on_input(self,key):
        if key=='q': quit()
        if key=='p': self.plot(self.profile)
        if key=='o': sysutil.showfile(self.outpath)
        return key

    def plot(self,P):
        options=['loss','|f|','|Af|','|f|/|Af|','|weights|','heatmaps','contours']
        plotoptions=self.run_subprocess(browse.Browse(onlyone=False,options=options))
        self.T.draw()
        if 'heatmaps' in plotoptions or 'contours' in plotoptions:
            I,Xp,Yp=P.plotslice
            M=jnp.quantile(jnp.abs(Yp),.9)
            fp=P.learner.eval(Xp)
            fp=fp*jnp.dot(fp,Yp)/jnp.dot(fp,fp)
            P.plotslice_=(I,Xp,Yp,fp)
        for o in ['heatmaps','contours']:
            match o:
                case 'heatmaps':
                    I,Xp,Yp,fp=P.plotslice_
                    M=jnp.quantile(jnp.abs(Yp),.9)
                    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(15,6))
                    handle0=ax0.pcolormesh(I,I,Yp,cmap='seismic',vmin=-M,vmax=M)
                    handle1=ax1.pcolormesh(I,I,fp,cmap='seismic',vmin=-M,vmax=M)
                    for m in [handle0,handle1]: m.set_edgecolor('face')
                    ct0=ax0.contour(I,I,Yp,levels=[-10,-1,-.1,0,.1,1,10],colors='k',linewidths=1)
                    ct1=ax1.contour(I,I,fp,levels=[-10,-1,-.1,0,.1,1,10],colors='k',linewidths=1)
                    for ax in [ax0,ax1]: ax.set_aspect('equal')
                case 'contours':
                    I,Xp,Yp,fp=P.plotslice_
                    fig,(ax)=plt.subplots(1,1,figsize=(6,6))
                    ct0=ax.contour(I,I,Yp,levels=[-10,-1,-.1,0,.1,1,10],colors='b',linewidths=2)
                    ct1=ax.contour(I,I,fp,levels=[-10,-1,-.1,0,.1,1,10],colors='r',linewidths=1)
                    ax.set_aspect('equal')

            outpath=os.path.join('plots',cfg.session.ID,'{}.pdf'.format(o))
            sysutil.savefig(outpath)
            sysutil.showfile(outpath)



    @staticmethod
    def getinstructions():
        return 'Press [p] to generate plots.\n'+\
            'Press [o] to open output folder.'+\
            '\n\nPress [b] to break from current task.\nPress [q] to quit. '

#    def log(self,msg):
#        super().log(msg)
#        self.act_on_input(cfg.getch(log=msg))
#        self.T.draw()

    def prepdisplay(self):
        instructions=self.getinstructions()

        self.dashboard=self.display
        self.T,self.learningdisplay=self.dashboard.vsplit(rlimits=[.8])
        self.L,self.R=self.T.hsplit()

        self.L.add(0,0,_display_._TextDisplay_(instructions))
        self.L.add(0,20,_display_._LogDisplay_(cfg.session,self.L.width,25,balign=False))
        self.L.vstack()

        self.infodisplay=self.R.add(0,0,_display_._TextDisplay_(''))

        self.T.arm()
        self.learningdisplay.arm()
        self.T.draw()

    def addlearningdisplay(self):
        k=100
        lossnames=self.losses.keys()
        smoothers={ln:numutil.RunningAvg(k=k) for ln in lossnames}
        display=self.learningdisplay.add(0,0,_display_._Display_())
        def bar_at(lossname,smoother,loss,k,pos):
            return [(0,3*pos+3,'{} (smoothed over {} iterations) {:.2E}'.format(lossname,k,smoother.update(loss))),\
            (0,3*pos+4,_display_.hiresbar(smoother.avg(),self.dashboard.width))]
            
        encode=lambda:[bar_at(ln,smoothers[ln],self.losses[ln][-1],k,pos=i) for i,ln in enumerate(lossnames)]
        display.encode=lambda: [(0,0,'{:,d}/{:,d} iterations'.format(self.getits(),self.profile.iterations))]\
                                +[s for l in encode() for s in l]+[(0,1,'')]

    @staticmethod
    def getslice(X,Y,rho):
        x=X[jnp.argmax(rho*Y**2)]
        I=jnp.arange(-3,3.03,.03)
        return slicethrough(x,I),I

    @classmethod
    def getprofiles(cls):
        return {'default':cls.getdefaultprofile}

    @classmethod
    def getdefaultprofile(cls,**kwargs):
        P=profile=super().getdefaultprofile()
        profile.n=5
        profile.d=2

        P.update(**kwargs)

        # training params

        profile.weight_decay=0.0
        profile.iterations=25000
        profile.batchsize=100

        profile.samples_train=10**5
        profile.samples_test=1000
        profile.evalblocksize=10**4

        profile._var_X_distr_=1
        profile._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
        profile.X_density=numutil.gen_nd_gaussian_density(var=profile._var_X_distr_)
        profile.X=profile._genX_(rnd.PRNGKey(0),profile.samples_train,profile.n,profile.d)
        profile.rho=profile.X_density(profile.X)
        profile.update(**kwargs)
        samplespipe=sampling.SamplesPipe(profile.X,profile.rho,minibatchsize=profile.batchsize)
        profile.X_rho_sampler=samplespipe.step

        def get_XYrho_sampler(Xrho_sampler,target):
            def newsampler():
                X,rho=Xrho_sampler()
                return X,target.eval(X),rho
            return newsampler

        profile.get_XYrho_sampler=get_XYrho_sampler

        def repeat(run,i):
            if i is None: return
            if i%100==0:
                log('{} iterations'.format(i))
            #run.traindata[i]['weights']=P.learner.weights
            #sysutil.save(P.learner.compress(),path=path.join(run.outpath,'data','learner'))
            #sysutil.save(run.traindata,path.join(run.outpath,'data','traindata'),echo=False)
            #sysutil.write('loss={:.2E} iterations={} n={} d={}'.format(loss,i,P.n,P.d),path.join(run.outpath,'metadata.txt'),mode='w')    
        
        P.repeat=repeat
        P.prep=lambda run:None
        P.finish=partial(repeat,i=None)

        P.parseinfo=lambda I:'\n'.join(['{}:{}'.format(k,v) for k,v in I.items()])
        P.info=dict(n=P.n,d=P.d)

        profile.update(**kwargs)
        return profile



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
    





####################################################################################################
# variations
####################################################################################################





class Run_statictarget(Run):
    processname='static_target'

    @classmethod
    def getXY(cls,P):
        if not 'Y' in P:
            cls.gettarget(P)
            P.Y=numutil.blockwise_eval(P.target,blocksize=P.evalblocksize,msg='preparing training data')(P.X)

    @classmethod
    def getdefaultprofile(cls,**kwargs):
        P=profile=super().getdefaultprofile(**kwargs)
        tracking.log('get X,Y')
        cls.getXY(profile)
        samplespipe=sampling.SamplesPipe(profile.X,profile.Y,profile.rho,minibatchsize=profile.batchsize)
        profile.sampler=samplespipe.step
        profile.learner=cls.getlearner(profile)
        profile.info.update(dict(learner=profile.learner.getinfo()))
        return profile






