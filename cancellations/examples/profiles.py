
from . import harmonicoscillator1d, harmonicoscillator2d, estimateobservables, unsupervised
from ..functions import examplefunctions as ef, functions
from ..functions.functions import Product, SingleparticleNN, ComposedFunction
from ..utilities import numutil, energy, tracking
from ..utilities.tracking import dotdict
import jax.numpy as jnp

def getprofiles(exname):
    profilegenerators=dict()
    exprofiles=profilegenerators
    match exname:


        case 'harmonicoscillator2d':
            exprofiles['default']=harmonicoscillator2d.Run.getdefaultprofile().butwith(weight_decay=.1)
            exprofiles['n=6']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.1)
#            exprofiles['n=7 wd=0']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=7,weight_decay=0)
#            exprofiles['n=7 wd=.1']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=7,weight_decay=.1)
#            exprofiles['n=7 wd=1']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=7,weight_decay=1.0)

            exprofiles['ASNN']=harmonicoscillator2d.Run.getdefaultprofile().butwith(\
                #
                weight_decay=.1,\
                n=5,\
                getlearner=lambda profile:
                    Product(functions.IsoGaussian(1.0),ComposedFunction(\
                    SingleparticleNN(**profile.learnerparams['SPNN']),\
                    functions.ASNN(**profile.learnerparams['ASNN']))),\
                #
                learnerparams=tracking.dotdict(\
                    SPNN=dotdict(widths=[2,25,10],activation='sp'),\
                    ASNN=dotdict(n=5,d=10,widths=[50,50,1],activation='sp')),\
                #
                minibatchsize=50,\
                evalblocksize=100\
                )
                
#        case 'harmonicoscillator1d':
#            exprofiles['default']=harmonicoscillator1d.Run.getdefaultprofile().butwith(weight_decay=.1)
#
#            exprofiles['no weight_decay']=harmonicoscillator1d.Run.getdefaultprofile().butwith(weight_decay=0)
#            exprofiles['weight_decay .1']=harmonicoscillator1d.Run.getdefaultprofile().butwith(weight_decay=.1)
#            exprofiles['weight_decay 1']=harmonicoscillator1d.Run.getdefaultprofile().butwith(weight_decay=1.0)
#            exprofiles['weight_decay 10']=harmonicoscillator1d.Run.getdefaultprofile().butwith(weight_decay=10.0)
#            exprofiles['weight growth']=harmonicoscillator1d.Run.getdefaultprofile().butwith(weight_decay=-.1)
#
#            exprofiles['layer normalization, no weight decay']=harmonicoscillator1d.Run.getdefaultprofile().butwith(\
#                layernormalization=(2.0,'batch'),\
#                weight_decay=0\
#                )
#
#            exprofiles['init weights small, no weight decay']=harmonicoscillator1d.Run.getdefaultprofile().butwith(\
#                initweight_coefficient=.1,\
#                weight_decay=0\
#                )
#
#            exprofiles['test']=harmonicoscillator1d.Run.getdefaultprofile().butwith(\
#                n=3,\
#                d=2,\
#                )
#

        case 'estimateobservables':

            # 1
            exprofiles['abstract (non-runnable)']=estimateobservables.Run.getdefaultprofile()

            # 2
            psi_descr=harmonicoscillator1d.gettarget(estimateobservables.Run.getdefaultprofile())
            psi=psi_descr.eval
            E_kin_local=numutil.forfixedparams(energy.genlocalkinetic)(psi)
            p_descr=functions.ComposedFunction(psi_descr,'square')
            profile=estimateobservables.Run.getdefaultprofile().butwith(\
                name='tgsamples',\
                p=p_descr.eval,\
                p_descr=p_descr,\
                psi_descr=psi_descr,\
                qpratio=lambda X:jnp.ones(X.shape[0],),\
                maxburnsteps=2500,\
                maxiterations=10**6,\
                observables={'V':lambda X:jnp.sum(X**2/2,axis=(-2,-1)),'K':E_kin_local},\
                burn_avg_of=1000)
            profile.trueenergies={k:ef.totalenergy(5)/2 for k in ['V','K']}

            exprofiles['true ground state']=profile


        case 'unsupervised':
            exprofiles['this example is under development']=unsupervised.Run.getdefaultprofile()


    #return {pname:lambda:pgen().butwith(profilename=pname) for pname,pgen in pgens.items()}
    return exprofiles







def get_test_fn_inputs():
    inputgenerators=dict()
    ig=inputgenerators


    ig['no inputs']=lambda: tracking.dotdict(args=[],kwargs=dict())


    def _():
        a='Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit,'
        b='sed do eiusmod \ntempor incididunt \nut labore et dolore \nmagna aliqua'
        return tracking.dotdict(args=[a+b,'do','p'],kwargs=dict())

    ig['lipsum, do, p']=_

    return ig