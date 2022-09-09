#
# nilin
# 
# 2022/7
#


from distutils.command.config import config
from re import I
import config as cfg
import functions
import display as disp
import exampletemplate
import jax
from functions import ComposedFunction,SingleparticleNN
import jax.random as rnd
import browse_runs
import util
import copy
import plottools as pt
import jax.numpy as jnp
import os
jax.config.update("jax_enable_x64", True)


profile=cfg.defaultrunprofile
profile.exname='example'
profile.instructions='To load and run with previously generated target function (including weights), run\
    \n\n>>python {}.py loadtarget'.format(profile.exname)
profile.outpath='outputs/{}/'.format(profile.runID)
cfg.outpath='outputs/{}/'.format(cfg.sessionID)

profile.targetparams={}
profile.targetchoice='ASNN1'

profile.learnerparams={}
profile.learnerchoice='backflow' 
#cfg.learnerchoice='backflow'; cfg.learnerparams=dict(activations=['tanh']*3)
#cfg.learnerchoice='ASNN2'


profile.n=5
profile.d=2

profile._X_distr_=lambda key,samples,n,d:rnd.uniform(key,(samples,n,d),minval=-1,maxval=1)
profile.envelope=jax.jit(lambda X:jnp.all(X**2<1,axis=(-2,-1)))

profile.trainingparams=dict\
(
weight_decay=0,
lossfn=util.SI_loss,
iterations=25000,
minibatchsize=None
)
profile.samples_train=10**5
profile.samples_test=1000
profile.evalblocksize=10**4




def pickexample(choice,n,d,**kw):
    n=cfg.currentprofile().n
    d=cfg.currentprofile().d

    match choice:
        # meant as target

        case 'hg':
            return ComposedFunction(functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen'),functions.Outputscaling())
        case 'gauss':
            return ComposedFunction(functions.Slater('parallelgaussians',n=n,d=d,mode='gen'),functions.Outputscaling())
        case 'ASNN1':
            m=cfg.providedefault(kw,m=10)
            return ComposedFunction(functions.ASNN(n=n,d=d,widths=['nd',m,m,1],activation='tanh'),functions.Outputscaling())

        # meant as learner

        case 'slaterNN':
            return ComposedFunction(\
                functions.Slater(SingleparticleNN(widths=[d,100,100,n],activation='tanh')),
                functions.OddNN(widths=[1,100,1],activation='leakyrelu'))

        case 'ASNN2': 
            d_=cfg.providedefault(kw,d_=10)
            m=cfg.providedefault(kw,m=10)

            return ComposedFunction(\
                SingleparticleNN(widths=[d,10,10,d_],activation='tanh'),\
                functions.ASNN(n=n,d=d_,widths=['nd',m,1],activation='leakyrelu'),\
                functions.OddNN(widths=[1,10,1],activation='leakyrelu'))

        case 'backflow':
            d_=cfg.providedefault(kw,d_=100)
            ndets=cfg.providedefault(kw,ndets=10)
            activations=cfg.providedefault(kw,activations=['leakyrelu','leakyrelu','leakyrelu'])

            return ComposedFunction(\
                SingleparticleNN(widths=[d,100,d_],activation=activations[0]),\
                functions.Backflow(widths=[d_,d_],activation=activations[1]),\
                functions.DetSum(n=n,d=d_,ndets=ndets),\
                functions.OddNN(widths=[1,100,1],activation=activations[2]))


def prep_and_run(profile):
    profile=cfg.currentprofile()
    cfg.log('imports done')

    #profile.retrieveparams(globals())
    n,d=profile.n,profile.d
    profile.X_distr=lambda key,samples:profile._X_distr_(key,samples,n,d)
    profile.unprocessed=cfg.Memory()
    info='runID: {}\n'.format(profile.runID)+'\n'*4; profile.run.trackcurrent('runinfo',info)


    if 'loadtarget' in cfg.cmdparams:
        try: path=cfg.loadtargetpath+'data/setup'
        except:
            loadpaths=browse_runs.pickfolders(msg='Choose target from previous run.\n'+\
                'Only the path in front of the arrow will be used.',\
                condition=lambda path:os.path.exists(path+'/data/setup'))
            path=loadpaths[0]+'data/setup'
        setupdata=cfg.Profile(cfg.load(path))

        target  =setupdata.target.restore()
        X_train =setupdata.X_train
        Y_train =setupdata.Y_train
        X_test  =setupdata.X_test
        Y_test  =setupdata.Y_test
        sections=setupdata.sections
        cfg.log('Loaded target and training data from '+path)

        info+='target\n\n{}'.format(cfg.indent(target.getinfo())); profile.run.trackcurrent('runinfo',info)


    else:
        target=pickexample(profile.targetchoice,n=n,d=d,**profile.targetparams)

        cfg.log('adjusting target weights')
        exampletemplate.adjustnorms(target,X=profile.genX(10000),iterations=250,learning_rate=.01)
        target=target.compose(functions.Flatten(sharpness=1))
        cfg.log('target initialized')

        info+='target\n\n{}'.format(cfg.indent(target.getinfo())); profile.run.trackcurrent('runinfo',info)

        X_train=profile.genX(profile.samples_train)
        profile.logcurrenttask('preparing training data')
        Y_train=target.eval(X_train,msg='preparing training data',blocksize=profile.evalblocksize)
        X_test=profile.genX(profile.samples_test)
        Y_test=target.eval(X_test,msg='preparing test data',blocksize=profile.evalblocksize)
        sections=pt.genCrossSections(target.eval)

    learner=pickexample(profile.learnerchoice,n=n,d=d,**profile.learnerparams)
    cfg.log('learner initialized')
    info+=4*'\n'+'learner\n\n{}'.format(cfg.indent(learner.getinfo())); profile.run.trackcurrent('runinfo',info)


    sourcedict=copy.copy(locals())
    setupdata={k:sourcedict[k] for k in ['X_train','Y_train','X_test','Y_test']}|\
        {'target':target.compress(),'learner':learner.compress(),'sections':sections}
    cfg.save(setupdata,profile.outpath+'data/setup')

    profile.register(\
        'target',\
        'learner',\
        'X_train',\
        'Y_train',\
        'X_test',\
        'Y_test',\
        'sections',\
        sourcedict=locals())

    profile.run.trackcurrent('runinfo',info)
    cfg.write(info,profile.outpath+'info.txt',mode='w')

    exampletemplate.inspect()
    exampletemplate.train(learner,X_train,Y_train,**profile.trainingparams)



def main(profile):
    profile.prepdashboard=exampletemplate.prepdashboard
    profile.act_on_input=exampletemplate.act_on_input
    import cdisplay
    cdisplay.run_in_display(prep_and_run,profile)

if __name__=='__main__':
    main(cfg.defaultrunprofile)
