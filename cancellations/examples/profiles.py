
from . import harmonicoscillator2d #, estimateobservables, unsupervised
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
            exprofiles['n=6 d=2']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.1)
            exprofiles['n=6 d=2 no weight decay']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=0)
            exprofiles['n=6 d=2 strong weight decay']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=1.)

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

            exprofiles['ASNN, no weight decay']=exprofiles['ASNN'].butwith(weight_decay=0)
                


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