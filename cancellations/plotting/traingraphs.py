from cancellations.utilities import textutil
from cancellations.functions import examplefunctions as ef, functions
from cancellations.utilities import numutil, tracking, sysutil
from cancellations.utilities.sysutil import maybe as maybe
import matplotlib.pyplot as plt
import jax.numpy as jnp

#
#

def getnorm(_f_,weights,X,Xdensity):
    Y=_f_(weights,X)
    squaresums=jnp.sum(Y**2,axis=Y.shape[1:])

    assert(squaresums.shape==Xdensity.shape)
    normalized=squaresums/Xdensity
    return jnp.average(normalized)


def graphdata(process,datapath):

    learner=sysutil.load(datapath+'data/learner').restore()
    traindata=sysutil.load(datapath+'data/traindata')
    X,Y,Xdensity=[sysutil.load(datapath+'data/setup')[k] for k in ['X_test','Y_test','Xdensity_test']]

    _psi_=learner._eval_
    fdescr,switchcounts=functions.switchtype(learner)
    _f_=fdescr._eval_

    assert(switchcounts==1)
    # if not 'ignoreAS' in sys.argv:
    #     testing.verify_antisymmetrization(learner.eval,fdescr.eval,X[:100])

    i_s,weightslist=tracking.extracthist(traindata,'i','weights')

    process.log('generating training graphs')

    Afs=[getnorm(_psi_,weights,X,Xdensity) for weights in weightslist]
    fs=[getnorm(_f_,weights,X,Xdensity) for weights in weightslist]

    weightnorms=[jnp.sqrt(numutil.recurseonleaves(weights,lambda A:jnp.sum(A**2) if A is not None else 0,sum)) for weights in weightslist]

    losses=[numutil.weighted_SI_loss(_psi_(weights,X),Y,Xdensity) for weights in weightslist]

    return i_s,losses,weightnorms,(Afs,fs)

def graph(process,datapath):

    i_s,losses,weightnorms,(Afs,fs)=graphdata(process,datapath)
    f_over_Af=[jnp.sqrt(f/Af) for f,Af in zip(fs,Afs)]


    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,10))

    ax1.plot(i_s,losses,'b',label='loss')
    ax1.set_yscale('log')
    ax1.grid(True,which='major',axis='y')
    ax1.legend()
    fig.suptitle(sysutil.maybe(lambda:'\nprofile name: '+sysutil.load(datapath+'data/setup')['profilename'],'')())

    ax2.plot(i_s,f_over_Af,'m:',label='|f|/|Af|')
    ax2.plot(i_s,fs,'r',label='|f|')
    ax2.plot(i_s,Afs,'b',label='|Af|')
    ax2.set_yscale('log')
    ax2.legend()

    ax3.plot(i_s,weightnorms,'r:',label='|W|')
    ax3.set_yscale('log')
    ax3.legend()

    sysutil.savefig(process.outpath+'train_graphs.pdf',fig=fig)

    fig,ax=plt.subplots()
    ax.scatter(losses,f_over_Af,s=6,color='b',marker='d')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    #ax.grid(True,which='both')
    ax.set_xlabel('loss (poor {} good)'.format(textutil.arrowright))
    ax.set_ylabel('|f|/|Af|')
    fig.suptitle(sysutil.maybe(lambda:'\nprofile name: '+sysutil.load(datapath+'data/setup')['profilename'],'')())

    sysutil.savefig(process.outpath+'normratio_vs_loss.pdf',fig=fig)


#dontpick



        #sysutil.showfile(process.outpath)

        #plt.show()

        #info=sysutil.readtextfile(runpath+'info.txt')
        #sysutil.write(info,batch.outpath+'info.txt')

#Run(**profile).run_as_main()
#