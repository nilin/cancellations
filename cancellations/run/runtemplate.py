import jax.numpy as jnp
import jax.random as rnd
import os
from os import path
from functools import partial, reduce
from cancellations.config import config as cfg, sysutil, tracking
from cancellations.functions import _functions_
from cancellations.functions._functions_ import ComposedFunction,SingleparticleNN,Product
from cancellations.run import sampling
from cancellations.utilities import numutil, textutil
from cancellations.config.tracking import dotdict
from cancellations.plotting import plotting
from cancellations.display import _display_
from cancellations.lossesandnorms import losses
from jax.tree_util import tree_map
import optax


class Run(_display_.Process):

    processname='example template'
    processtype='runs'

    def execprocess(run:_display_.Process):

        run.prepdisplay()

        P=run.profile
        run.info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=run.info

        run.info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(P.learner.getinfo()))#; run.infodisplay.msg=info
        run.infodisplay.msg=run.info
        run.T.draw()

        run.traindata=dict()
        run.info+=10*'\n'+str(run.profile) 
        sysutil.write(run.info,path.join(run.outpath,'info.txt'),mode='w')

        regsched=tracking.Scheduler(range(0,P.iterations+25,25))
        run.losses={ln:None for ln in P.lossnames}
        run.addlearningdisplay()

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        opt=optax.adamw(learning_rate=.01,weight_decay=P.weight_decay)
        state=opt.init(P.learner.weights)
        #run.learner.weights=listform(run.learner.weights)

        for i in range(P.iterations+1):

            X,*Ys=P.sampler()

            Ls,Gs=zip(*[lossgrad(P.learner.weights,X,*Ys) for lossgrad in P.lossgrads])
            for i,(ln,loss) in enumerate(zip(P.lossnames,Ls)):
                run.losses[ln]=loss
            run.traindata[i]={ln:loss for ln,loss in zip(P.lossnames,Ls)}
            grad=sumgrads(rescale(P.lossweights,Gs))

            updates,state=opt.update(grad,state,P.learner.weights)
            P.learner.weights=optax.apply_updates(P.learner.weights,updates)

            run.its=i
            if regsched.activate(i):
                run.traindata[i]['weights']=P.learner.weights
                sysutil.save(P.learner.compress(),path=path.join(run.outpath,'data','learner'))
                sysutil.save(run.traindata,path.join(run.outpath,'data','traindata'),echo=False)
                #sysutil.write('loss={:.2E} iterations={} n={} d={}'.format(loss,i,P.n,P.d),path.join(run.outpath,'metadata.txt'),mode='w')    

            if stopwatch1.tick_after(.05):
                run.learningdisplay.draw()

            if stopwatch2.tick_after(.5):
                if run.act_on_input(cfg.getch(run.getinstructions))=='b': break
        
        plotting.allplots(run)
        return run.learner

    def act_on_input(self,key):
        if key=='q': quit()
        if key=='p': self.plot()
        if key=='o': sysutil.showfile(self.outpath)
        return key

    def plot(self):
        self.log('')
        pass

    @staticmethod
    def getinstructions():
        return 'Press [p] to generate plots.\n'+\
            'Press [o] to open output folder.'+\
            '\n\nPress [b] to break from current task.\nPress [q] to quit. '

    def log(self,msg):
        super().log(msg)
        self.act_on_input(cfg.getch(log=msg))
        self.T.draw()

    def prepdisplay(self):
        instructions=self.getinstructions()

        self.dashboard=self.display
        self.T,self.learningdisplay=self.dashboard.vsplit(rlimits=[.8])
        self.L,self.R=self.T.hsplit()

        self.L.add(0,0,_display_._TextDisplay_(instructions))
        self.L.add(0,20,_display_._LogDisplay_(self,self.L.width,20))
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
            
        encode=lambda:[bar_at(ln,smoothers[ln],self.losses[ln],k,pos=i) for i,ln in enumerate(lossnames)]
        display.encode=lambda: [(0,0,'{:,d}/{:,d} iterations'.format(self.its,self.profile.iterations))]\
                                +[s for l in encode() for s in l]+[(0,1,'')]

    @classmethod
    def getprofiles(cls):
        return {'default':cls.getdefaultprofile}

    @staticmethod
    def getdefaultprofile(**kwargs):
        profile=tracking.Profile()
        profile.info=''
        profile.n=5
        profile.d=2
        profile.update(**kwargs)

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
        samplespipe=sampling.SamplesPipe(profile.X,minibatchsize=profile.batchsize)
        profile.Xsampler=samplespipe.step

        def getXYsampler(Xsampler,target):
            def newsampler():
                (X,)=Xsampler()
                return X,target.eval(X)
            return newsampler

        profile.getXYsampler=getXYsampler
        return profile

def rescale(cs,Gs):
    return [tree_map(lambda A:c*A,G) for c,G in zip(cs,Gs)]

def sumgrads(Gs):
    add=lambda *As: reduce(jnp.add,As)
    return tree_map(add,*Gs)
    


###### example ######

def getlearner_example(profile):

    P=profile
    profile.learnerparams=tracking.dotdict(\
        SPNN=dotdict(widths=[profile.d,25,25],activation='sp'),\
        backflow=dotdict(widths=[25,25,25],activation='sp'),\
        dets=dotdict(d=25,ndets=P.ndets),)
        #'OddNN':dict(widths=[25,1],activation='sp')

    return Product(_functions_.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        _functions_.Backflow(**profile.learnerparams['backflow']),\
        _functions_.Dets(n=profile.n,**profile.learnerparams['dets']),\
        _functions_.Sum()\
        ))
