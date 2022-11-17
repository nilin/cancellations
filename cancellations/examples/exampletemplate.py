#
# nilin
# 
# 2022/7
#


from re import I
import jax
import jax.numpy as jnp
import jax.random as rnd
from ..functions import examplefunctions as ef, examplefunctions3d, functions
from ..learning import learning
from ..functions.functions import ComposedFunction,SingleparticleNN,Product
from ..utilities import config as cfg, numutil, tracking, sysutil, textutil, sampling, setup
from ..utilities.tracking import dotdict
from ..plotting import plotting
from ..display import _display_
from . import plottools as pt
from . import exampleutil
import os
import math
from functools import partial
from . import losses





#dontpick


class Run(_display_.Process):

    processname='example template'

    def execprocess(run:_display_.Process):

        run.prepdisplay()
        run.log('imports done')


        P=run.profile
        info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=info

        # make this temporary
        if 'layernormalization' in P.keys():
            cfg.layernormalization=P.layernormalization 
        if 'initweight_coefficient' in P.keys():
            cfg.initweight_coefficient=P.initweight_coefficient

        run.log('gen target')
        run.genX=lambda nsamples: P._genX_(run.nextkey(),nsamples,P.n,P.d)

        run.target=P.gettarget(P,run)
        info+='target\n\n{}'.format(textutil.indent(run.target.getinfo()))
        run.infodisplay.msg=info
        run.T.draw()
        sysutil.save(run.target.compress(),path=run.outpath+'data/target')

        run.log('gen samples')

        run.X_train=run.genX(P.samples_train)

        run.log('preparing training data')

        run.Y_train=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing training data')(run.X_train)
        run.X_test=run.genX(P.samples_test)
        run.Y_test=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing test data')(run.X_test)
        r=P.plotrange
        #run.sections=pt.genCrossSections(numutil.blockwise_eval(run.target,blocksize=P.evalblocksize),interval=jnp.arange(-r,r,r/50))

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
            #sections=run.sections,\
            profilename=P.profilename\
            )
        sysutil.save(setupdata,run.outpath+'data/setup')

        run.traindata=[]

        info+=10*'\n'+str(run.profile) 
        sysutil.write(info,run.outpath+'info.txt',mode='w')

        run.log('gen lossgrad and learner')
        #train
        run.lossgradobj=P.initlossgrad(run.learner._eval_,P.X_density)
        run.lossgrad=run.lossgradobj._eval_

        run.sampler=sampling.SamplesPipe(run.X_train,run.Y_train,minibatchsize=P.minibatchsize)
        run.trainer=learning.Trainer(run.lossgrad,run.learner,run.sampler,\
            **{k:P[k] for k in ['weight_decay','iterations']}) 


        regsched=tracking.Scheduler(range(0,P.iterations+100,100))
        run.addlearningdisplay()

        run.log('data type (32 or 64): {}'.format(run.learner.eval(run.X_train[100:]).dtype))

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        for i in range(P.iterations+1):

            loss=run.trainer.step()
            run.loss.val=loss
            run.samplesdone=i*P.minibatchsize
            run.learningdisplay.draw()

            run.traindata.append(dict(loss=loss,i=i))

            if regsched.activate(i):
                run.traindata.append(dict(weights=run.learner.weights,i=i))
                sysutil.save(run.learner.compress(),path=run.outpath+'data/learner')
                sysutil.save(run.traindata,run.outpath+'data/traindata',echo=False)
                sysutil.write('loss={:.2E} iterations={} n={} d={}'.format(loss,i,P.n,P.d),run.outpath+'metadata.txt',mode='w')    

            if stopwatch1.tick_after(.05):
                run.learningdisplay.draw()

            if stopwatch2.tick_after(.5):
                if P.act_on_input(setup.getch(getinstructions),run)=='b': break
        
        plotting.allplots(run)
        return run.learner

    def log(self,msg):
        super().log(msg)
        self.profile.act_on_input(setup.getch(log=msg),self)
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
        smoother1=numutil.RunningAvg(k=1)
        smoother2=numutil.RunningAvg(k=100)
        display=self.learningdisplay.add(0,0,_display_._Display_())
        display.encode=lambda: [\
            (0,0,'training loss (avg over 100 minibatches) {:.2E}'.format(smoother2.update(self.loss.val))),\
            (0,1,_display_.hiresbar(smoother2.avg(),self.dashboard.width)),\
            #
            #(0,1,_display_.hiresbar(self.loss.val,self.dashboard.width)),\
            #_display_.hiresTICK(smoother1.update(self.loss.val),self.dashboard.width,y=1),\
            #_display_.hirestick(smoother1.avg(),self.dashboard.width,y=2),\
            #
            (0,4,_display_.thinbar(self.loss.val,self.dashboard.width)),\
            (0,3,'training loss (non-smoothed) {:.2E}'.format(self.loss.val)),\
            (0,8,'{:,d}/{:,d} samples'.format(self.samplesdone,self.profile.minibatchsize*self.profile.iterations))
            ]
        



    @classmethod
    def getdefaultprofile(cls):
        profile=tracking.Profile()
        profile.instructions=''

        profile.gettarget=gettarget
        profile.getlearner=getlearner
        profile.initlossgrad=losses.Lossgrad_SI

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
        #lambda X:\
        #    jnp.exp(-jnp.sum(X**2/(2*profile._var_X_distr_),axis=(-2,-1)))\
        #    /(2*math.pi*profile._var_X_distr_)**(X.shape[-2]*X.shape[-1]/2)

        # training params

        profile.weight_decay=0
        profile.iterations=25000
        profile.minibatchsize=100

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
    return Product(functions.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        #functions.Backflow(**profile.learnerparams['backflow']),\
        functions.Dets(n=profile.n,**profile.learnerparams['dets']),\
        functions.Sum()\
        ))




