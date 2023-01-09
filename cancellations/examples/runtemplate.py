import jax.numpy as jnp
import jax.random as rnd
import os
from os import path
from functools import partial
from cancellations.config import config as cfg, sysutil, tracking
from cancellations.functions import _functions_
from cancellations.learning import learning
from cancellations.functions._functions_ import ComposedFunction,SingleparticleNN,Product
from cancellations.utilities import numutil, textutil, sampling
from cancellations.config.tracking import dotdict
from cancellations.plotting import plotting
from cancellations.display import _display_
from cancellations.lossesandnorms import losses


#dontpick


class Run(_display_.Process):

    processname='example template'
    processtype='runs'

    def execprocess(run:_display_.Process):

        run.prepdisplay()
        run.log('imports done')

        P=run.profile
        info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=info

        run.log('gen target')
        run.genX=lambda nsamples: P._genX_(run.nextkey(),nsamples,P.n,P.d)

        run.target=P.gettarget(P,run)
        info+='target\n\n{}'.format(textutil.indent(run.target.getinfo()))
        run.infodisplay.msg=info
        run.T.draw()
        sysutil.save(run.target.compress(),path=os.path.join(run.outpath,'data/target'))

        run.log('gen samples')

        run.X_train=run.genX(P.samples_train)

        run.log('preparing training data')

        run.Y_train=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing training data')(run.X_train)
        run.X_test=run.genX(P.samples_test)
        run.Y_test=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing test data')(run.X_test)
        r=P.plotrange

        run.log('gen learner')

        run.learner=P.getlearner(P)
        info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo()))#; run.infodisplay.msg=info
        run.infodisplay.msg=info
        run.T.draw()

        run.log('save setup data')

        setupdata=dict(\
            target=run.target.compress(),\
            learner=run.learner.compress(),\
            #
            X_train=run.X_train, Y_train=run.Y_train, Xdensity_train=P.X_density(run.X_train),\
            X_test=run.X_test,Y_test=run.Y_test, Xdensity_test=P.X_density(run.X_test),\
            profilename=P.profilename\
            )
        sysutil.save(setupdata,os.path.join(run.outpath,'data/setup'))

        run.traindata=[]

        info+=10*'\n'+str(run.profile) 
        sysutil.write(info,path.join(run.outpath,'info.txt'),mode='w')

        run.log('gen lossgrad and learner')
        #train

        run.sampler=sampling.SamplesPipe(run.X_train,run.Y_train,minibatchsize=P.batchsize)

        run.lossgrads=[]
        run.trainers=[]
        for init in P.initlossgrads:
            run.lossgradobj=init(run.learner,P.X_density)
            lossgrad=run.lossgradobj._eval_
            run.lossgrads.append(lossgrad)
            run.trainers.append(learning.Trainer(lossgrad,run.learner,run.sampler,\
                **{k:P[k] for k in ['weight_decay','iterations']}))


        regsched=tracking.Scheduler(range(0,P.iterations+25,25))
        run.addlearningdisplay()

        run.log('data type (32 or 64): {}'.format(run.learner.eval(run.X_train[100:]).dtype))

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        for i in range(P.iterations+1):

            for name,dt,trainer in zip(P.lossnames,P.lossperiods,run.trainers):
                if i%dt==0:
                    loss=trainer.step()
                    if name=='loss':
                        run.loss.val=loss

                run.traindata.append(dict(name=loss,i=i))

            run.its=i
            run.learningdisplay.draw()

            if regsched.activate(i):
                run.traindata.append(dict(weights=run.learner.weights,i=i))
                sysutil.save(run.learner.compress(),path=path.join(run.outpath,'data','learner'))
                sysutil.save(run.traindata,path.join(run.outpath,'data','traindata'),echo=False)
                sysutil.write('loss={:.2E} iterations={} n={} d={}'.format(loss,i,P.n,P.d),path.join(run.outpath,'metadata.txt'),mode='w')    

            if stopwatch1.tick_after(.05):
                run.learningdisplay.draw()

            if stopwatch2.tick_after(.5):
                if P.act_on_input(cfg.getch(getinstructions),run)=='b': break
        
        plotting.allplots(run)
        return run.learner

    def log(self,msg):
        super().log(msg)
        self.profile.act_on_input(cfg.getch(log=msg),self)
        self.T.draw()

    def prepdisplay(self):
        instructions=getinstructions()

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
        self.loss=tracking.Pointer()
        smoother2=numutil.RunningAvg(k=100)
        display=self.learningdisplay.add(0,0,_display_._Display_())
        display.encode=lambda: [\
            (0,0,'training loss (avg over 100 iterations) {:.2E}'.format(smoother2.update(self.loss.val))),\
            (0,1,_display_.hiresbar(smoother2.avg(),self.dashboard.width)),\
            #
            (0,4,_display_.thinbar(self.loss.val,self.dashboard.width)),\
            (0,3,'training loss (non-smoothed) {:.2E}'.format(self.loss.val)),\
            (0,8,'{:,d}/{:,d} iterations'.format(self.its,self.profile.iterations))
            ]
        



    @classmethod
    def getdefaultprofile(cls):
        profile=tracking.Profile()
        profile.instructions=''

        profile.gettarget=gettarget
        profile.getlearner=getlearner

        #losses
        profile.initlossgrads=[losses.Lossgrad_SI]
        profile.lossnames=['loss']
        profile.lossperiods=[1]

        def act_on_input(key,process):
            if key=='q': quit()
            if key=='p': plotting.allplots(process)
            if key=='o': sysutil.showfile(process.outpath)
            return key

        profile.act_on_input=act_on_input

        profile.n=5
        profile.d=2

        profile.learnerparams=tracking.dotdict(\
            SPNN=dotdict(widths=[profile.d,25,25],activation='sp'),\
            #backflow=dotdict(widths=[],activation='sp'),\
            dets=dotdict(d=25,ndets=25),)
            #'OddNN':dict(widths=[25,1],activation='sp')

        profile._var_X_distr_=1
        profile._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
        profile.X_density=numutil.gen_nd_gaussian_density(var=profile._var_X_distr_)

        # training params

        profile.weight_decay=0
        profile.iterations=25000
        profile.batchsize=100

        profile.samples_train=10**5
        profile.samples_test=1000
        profile.evalblocksize=10**4

        profile.adjusttargetsamples=10000
        profile.adjusttargetiterations=250

        profile.plotrange=5

        return profile


    @classmethod
    def getprofiles(cls):
        profiles=dict()
        default=cls.getdefaultprofile().butwith(n=6,weight_decay=.1)
        profiles['n=6 d=2 SI']=default
        profiles['n=6 d=2 non-SI']=default.butwith(initlossgrad=losses.Lossgrad_nonSI)
        profiles['n=6 d=2 unbiased loss']=default.butwith(\
            initlossgrad=partial(losses.Lossgrad_unbiased,10))
        return profiles

def getinstructions():
    return 'Press [p] to generate plots.\n'+\
        'Press [o] to open output folder.'+\
        '\n\nPress [b] to break from current task.\nPress [q] to quit. '



def gettarget(P,run):
    raise NotImplementedError

def getlearner(profile):
    #return Product(functions.ScaleFactor(),functions.IsoGaussian(1.0),ComposedFunction(\
    return Product(_functions_.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        #functions.Backflow(**profile.learnerparams['backflow']),\
        _functions_.Dets(n=profile.n,**profile.learnerparams['dets']),\
        _functions_.Sum()\
        ))




