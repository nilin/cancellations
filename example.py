#
# nilin
# 
# 2022/7
#


import config as cfg
import functions
import display as disp
from config import session
import exampletemplate
import jax
from functions import ComposedFunction,SingleparticleNN
import jax.random as rnd
import browse_runs
import util
import copy
import plottools as pt
from config import defaultrunprofile as dprof
import os
jax.config.update("jax_enable_x64", True)



dprof.exname='example'
dprof.instructions='To load and run with previously generated target function (including weights), run\
    \n\n>>python {}.py loadtarget'.format(dprof.exname)
dprof.outpath='outputs/{}/'.format(dprof.runID)
cfg.outpath='outputs/{}/'.format(cfg.sessionID)

dprof.targetparams={}
dprof.targetchoice='ASNN1'

dprof.learnerparams={}
dprof.learnerchoice='backflow' 
#cfg.learnerchoice='backflow'; cfg.learnerparams=dict(activations=['tanh']*3)
#cfg.learnerchoice='ASNN2'


n=5
d=2

dprof.trainingparams=dict\
(
weight_decay=0,
lossfn=util.SI_loss,
iterations=25000,
minibatchsize=None
)
dprof.samples_train=100000
dprof.samples_test=1000




def pickexample(choice,n,d,**kw):

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


def prep_and_run(runprofile=None):
    if runprofile is not None: cfg._currentprofile_=runprofile
    cprof=cfg.currentprofile()
    cfg.log('imports done')

    cprof.retrieveparams(globals())
    cprof.register('n','d',sourcedict=globals())
    cprof.X_distr=lambda key,samples:rnd.uniform(key,(samples,n,d),minval=-1,maxval=1)
    cprof.unprocessed=cfg.Memory()
    info='runID: {}\n'.format(cprof.runID)+'\n'*4; cprof.run.trackcurrent('runinfo',info)


    if 'loadtarget' in cfg.cmdparams:
        try: path=cfg.loadtargetpath+'data/setup'
        except:
            path=browse_runs.pickfolders(multiple=False,msg='Choose target from previous run.',\
                condition=lambda path:os.path.exists(path+'/data/setup'))+'data/setup'
        setupdata=cfg.load(path)

        target  =setupdata['target'].restore()
        X_train =setupdata['X_train']
        Y_train =setupdata['Y_train']
        X_test  =setupdata['X_test']
        Y_test  =setupdata['Y_test']
        sections=setupdata['sections']
        cfg.log('Loaded target and training data from '+path)

        info+='target\n\n{}'.format(cfg.indent(target.getinfo())); cprof.run.trackcurrent('runinfo',info)


    else:
        target=pickexample(cprof.targetchoice,n=n,d=d,**cprof.targetparams)

        cfg.log('adjusting target weights')
        exampletemplate.adjustnorms(target,X=cprof.genX(10000),iterations=250,learning_rate=.01)
        target=target.compose(functions.Flatten(sharpness=1))
        cfg.log('target initialized')

        info+='target\n\n{}'.format(cfg.indent(target.getinfo())); cprof.run.trackcurrent('runinfo',info)

        X_train=cprof.genX(cprof.samples_train)
        cprof.logcurrenttask('preparing training data')
        Y_train=target.eval(X_train,msg='preparing training data')
        X_test=cprof.genX(cprof.samples_test)
        Y_test=target.eval(X_test,msg='preparing test data')
        sections=pt.genCrossSections(target.eval)

    learner=pickexample(cprof.learnerchoice,n=n,d=d,**cprof.learnerparams)
    cfg.log('learner initialized')
    info+=4*'\n'+'learner\n\n{}'.format(cfg.indent(learner.getinfo())); cprof.run.trackcurrent('runinfo',info)


    sourcedict=copy.copy(locals())
    setupdata={k:sourcedict[k] for k in ['X_train','Y_train','X_test','Y_test']}|\
        {'target':target.compress(),'learner':learner.compress(),'sections':sections}
    cfg.save(setupdata,cprof.outpath+'data/setup')

    cprof.register(\
        'target',\
        'learner',\
        'X_train',\
        'Y_train',\
        'X_test',\
        'Y_test',\
        'sections',\
        sourcedict=locals())

    cprof.run.trackcurrent('runinfo',info)
    cfg.write(info,cprof.outpath+'info.txt',mode='w')

    exampletemplate.inspect()
    exampletemplate.train(learner,X_train,Y_train,**cprof.trainingparams)

if __name__=='__main__':
    cfg.currentprofile().prepdashboard=exampletemplate.prepdashboard
    cfg.currentprofile().act_on_input=exampletemplate.act_on_input
    import cdisplay
    cdisplay.run_in_display(prep_and_run)
