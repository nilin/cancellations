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
        run.log('imports done')

        P=run.profile
        run.info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=run.info

        run.sampler=run.getsampler(P)

        run.log('gen learner')
        run.learner=P.getlearner(P)
        run.info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo()))#; run.infodisplay.msg=info
        run.infodisplay.msg=run.info
        run.T.draw()

        run.log('save setup data')

        run.traindata=dict()
        run.info+=10*'\n'+str(run.profile) 
        sysutil.write(run.info,path.join(run.outpath,'info.txt'),mode='w')
        run.log('gen lossgrad and learner')
        #train

        run.lossgrads=[]
        for init in P.initlossgrads:
            run.lossgradobj=init(run.learner,P.X_density)
            lossgrad=run.lossgradobj._eval_
            run.lossgrads.append(lossgrad)

        regsched=tracking.Scheduler(range(0,P.iterations+25,25))
        run.losses={ln:None for ln in P.lossnames}
        run.addlearningdisplay()

        run.log('data type (32 or 64): {}'.format(run.learner.eval(run.sampler.step(run.learner.weights)[0]).dtype))

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        opt=optax.adamw(learning_rate=.01,weight_decay=P.weight_decay)
        state=opt.init(run.learner.weights)
        #run.learner.weights=listform(run.learner.weights)

        for i in range(P.iterations+1):

            X,*Ys=run.sampler.step(run.learner.weights)

            Ls,Gs=zip(*[lossgrad(run.learner.weights,X,*Ys) for lossgrad in run.lossgrads])
            for i,(ln,loss) in enumerate(zip(P.lossnames,Ls)):
                run.losses[ln]=loss
            run.traindata[i]={ln:loss for ln,loss in zip(P.lossnames,Ls)}
            grad=sumgrads(rescale(P.lossweights,Gs))

            updates,state=opt.update(grad,state,run.learner.weights)
            run.learner.weights=optax.apply_updates(run.learner.weights,updates)

            run.its=i
            if regsched.activate(i):
                run.traindata[i]['weights']=run.learner.weights
                sysutil.save(run.learner.compress(),path=path.join(run.outpath,'data','learner'))
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
        if key=='p': self.allplots()
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
        k=10
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
        default=cls.getdefaultprofile()
        return {'default':default}


def rescale(cs,Gs):
    return [tree_map(lambda A:c*A,G) for c,G in zip(cs,Gs)]

def sumgrads(Gs):
    add=lambda *As: reduce(jnp.add,As)
    return tree_map(add,*Gs)
    
