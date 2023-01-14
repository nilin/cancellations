import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import os
from cancellations.config import config as cfg, sysutil, tracking
from cancellations.utilities import numutil as mathutil, textutil, numutil
from cancellations.functions import _functions_
from cancellations.testing import testing



def testantisymmetry(target,learner,X):
    cfg.logcurrenttask('verifying antisymmetry of target')
    testing.verify_antisymmetric(target.eval,X[:100])
    cfg.logcurrenttask('verifying antisymmetry of learner')
    testing.verify_antisymmetric(learner.eval,X[:100])
    cfg.clearcurrenttask()
    return True


def adjustnorms(Afdescr,X,iterations=500,**learningparams):
    run=cfg.currentprocess()
    Af=Afdescr._eval_
    f=_functions_.switchtype(Afdescr)._eval_
    normratio=jax.jit(lambda weights,X:mathutil.norm(f(weights,X))/mathutil.norm(Af(weights,X)))
    weights=Afdescr.weights

    cfg.log('|f|/|Af|={:.3f}, |Af|={:.3f} before adjustment'.format(\
        normratio(weights,X[:1000]),mathutil.norm(Af(weights,X[:1000]))))

    @jax.jit
    def directloss(params,Y):
        Af_norm=mathutil.norm(Af(params,Y))
        f_norm=mathutil.norm(f(params,Y))
        normloss=jnp.abs(jnp.log(Af_norm))
        ratioloss=jnp.log(f_norm/Af_norm)
        return normloss+ratioloss


    _,key1=run.display.column1.add(disp.NumberPrint('target |f|/|Af|',msg='\n\n|f|/|Af|={:.3f} (objective: decrease)'))
    _,key2=run.display.column1.add(disp.RplusBar('target |f|/|Af|'))
    _,key3=run.display.column1.add(disp.NumberPrint('target |Af|',msg='\n|Af|={:.3f} (objective: approach 1)'))
    _,key4=run.display.column1.add(disp.RplusBar('target |Af|'))
    
    for i in range(iterations):
        trainer.step()
        run.trackcurrent('target |Af|',mathutil.norm(Af(trainer.learner.weights,X[:100])))
        run.trackcurrent('target |f|/|Af|',normratio(trainer.learner.weights,X[:100]))
        if tracking.stopwatch.tick_after(.05) and cfg.act_on_input(cfg.checkforinput())=='b':break

    run.display.column1.delkeys(key1,key2,key3,key4)

    weights=trainer.learner.weights
    cfg.log('|f|/|Af|={:.3f}, |Af|={:.3f} after adjustment'.format(\
        normratio(weights,X[:1000]),mathutil.norm(Af(weights,X[:1000]))))
    #Afdescr.weights=weights
    #return weights


# info
####################################################################################################


def info(separator=' | '):
    run=cfg.currentprocess()
    P=run.profile
    return 'n={}, target: {}{}learner: {}'.format(P.n,\
        run.target.richtypename(),separator,run.learner.richtypename())

def INFO(separator='\n\n',width=100):
    run=cfg.currentprocess()
    targetinfo='target\n\n{}'.format(textutil.indent(run.target.getinfo()))
    learnerinfo='learner\n\n{}'.format(textutil.indent(run.learner.getinfo()))
    return disp.wraptext(targetinfo+'\n'*4+learnerinfo)








# plots
####################################################################################################

# learning plots

def process_snapshot_0(processed,f,X,Y,i):
    #processed.addcontext('minibatchnumber',i)
    processed.remember('Af norm',jnp.average(f(X[:100])**2),minibatchnumber=i)
    processed.remember('test loss',mathutil.SI_loss(f(X),Y),minibatchnumber=i)

def plotexample_0(unprocessed,processed):
    plt.close('all')

    fig,(ax0,ax1)=plt.subplots(2)
    fig.suptitle('test loss '+info())

    ax0.plot(*mathutil.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
    ax0.legend()
    ax0.set_ylim(bottom=0,top=1)
    ax0.grid(True,which='major',ls='-',axis='y')
    ax0.grid(True,which='minor',ls=':',axis='y')

    ax1.plot(*mathutil.swap(*processed.gethist('test loss','minibatchnumber')),'r-',label='test loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True,which='major',ls='-',axis='y')
    ax1.grid(True,which='minor',ls=':',axis='y')
    sysutil.savefig(os.path.join(cfg.outpath,'losses.pdf'),fig=fig)

    fig,ax=plt.subplots()
    ax.set_title('performance '+info())
    I,t=unprocessed.gethistbytime('minibatchnumber')
    ax.plot(t,I)
    ax.set_xlabel('time')
    ax.set_ylabel('minibatch')
    sysutil.savefig(os.path.join(cfg.outpath,'performance.pdf'),fig=fig)


process_snapshot=process_snapshot_0
plotexample=plotexample_0

def processandplot(unprocessed,pfunc,X,Y,process_snapshot_fn=None,plotexample_fn=None):

    pfunc=pfunc.getemptyclone()
    if process_snapshot_fn is None: process_snapshot_fn=process_snapshot
    if plotexample_fn is None: plotexample_fn=plotexample
    processed=cfg.Memory()

    weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')
    for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):

        if cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))=='b': break
        process_snapshot(processed,mathutil.fixparams(pfunc._eval_,weights),X,Y,i)        

    plotexample(unprocessed,processed)
    cfg.clearcurrenttask()
    return processed

def lplot():
    run=cfg.currentprocess()
    processandplot(run.unprocessed,run.learner,run.X_test,run.Y_test)


# function plots

def plotfunctions(sections,f,figtitle,path):
    plt.close('all')
    for fignum,section in enumerate(sections):
        fig=section.plot(f)
        if cfg.trackcurrenttask('generating function plots',(fignum+1)/len(sections))=='b': break
        fig.suptitle(figtitle+'\n\n'+section.info)
        sysutil.savefig('{} {}.pdf'.format(path,fignum),fig=fig)
    cfg.clearcurrenttask()

def fplot():
    run=cfg.currentprocess()
    figtitle=info(separator='\n')
    figpath=os.path.join(run.outpath,int(run.unprocessed.getval('minibatchnumber'))+' minibatches')

#    C=numutil.norm(run.target.eval(run.X_train[:1000]))/numutil.norm(run.learner.eval(run.X_train[:1000]))
#    f=lambda X:run.learner.eval(X)*C

    f=numutil.closest_multiple(run.learner.eval,run.X_train[:500],run.target.eval(run.X_train[:500]))

    plotfunctions(run.sections,f,figtitle,figpath)
    #plotfunctions(run.sections,run.learner.eval,figtitle,figpath)


# dashboard
####################################################################################################

def act_on_input(key):
    if key=='q': quit()
    #if key=='b': tracking.breaker.breaknow()
    if key=='l': lplot()
    if key=='f': fplot()
    if key=='o': sysutil.showfile(cfg.currentprocess().outpath)
    return key


