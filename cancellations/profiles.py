
from cancellations.examples import harmonicoscillator2d #, estimateobservables, unsupervised
from cancellations.functions import examplefunctions as ef, functions
from cancellations.functions.functions import Product, SingleparticleNN, ComposedFunction
from cancellations.utilities import numutil, energy, tracking
from cancellations.utilities.tracking import dotdict
import jax.numpy as jnp

def getprofiles(exname):
    exprofiles=dict()
    match exname:


        case 'harmonicoscillator2d':
            defaultprofile=harmonicoscillator2d.Run.getdefaultprofile()

            exprofiles['n=6 d=2 SI']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.1)
            exprofiles['n=6 d=2 NON-SI']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.1,\
                    gen_lossgrad=harmonicoscillator2d.gen_nonSI_lossgrad
                )
            exprofiles['n=6 d=2 NON-SI overall scaling DOF']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.1,\
                getlearner=lambda profile:\
                    Product(functions.ScaleFactor(),functions.IsoGaussian(1.0),ComposedFunction(\
                        SingleparticleNN(**profile.learnerparams['SPNN']),\
                        functions.Dets(n=profile.n,**profile.learnerparams['dets']),\
                        functions.Sum())),\
                    gen_lossgrad=harmonicoscillator2d.gen_nonSI_lossgrad
                )

            exprofiles['n=6 d=2 strong weight decay']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=1.)
            exprofiles['n=6 d=2 no weight decay']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=0)

            exprofiles['n=6 d=2 tanh']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.0,
                learnerparams=tracking.dotdict(\
                SPNN=dotdict(widths=[defaultprofile.d,100,100],activation='tanh'),\
                dets=dotdict(d=100,ndets=25),))

            exprofiles['n=6 d=2 lrelu']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=6,weight_decay=.0,
                learnerparams=tracking.dotdict(\
                SPNN=dotdict(widths=[defaultprofile.d,100,100],activation='lrelu'),\
                dets=dotdict(d=100,ndets=25),))

            exprofiles['n=7 d=2 weight decay .1']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=7,weight_decay=.1)
            exprofiles['n=7 d=2 weight decay .1, large batch']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=7,weight_decay=.1,minibatchsize=500)

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
                
            exprofiles['quicktest']=harmonicoscillator2d.Run.getdefaultprofile().butwith(n=4,iterations=1000,samples_train=10**4)


    return exprofiles







def get_test_fn_inputs():
    inputprofiles=dict()
    inputprofiles['no inputs']=lambda: tracking.dotdict(args=[],kwargs=dict())


    s='Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit,'+\
        'sed do eiusmod \ntempor incididunt \nut labore et dolore \nmagna aliqua'

    inputprofiles['lipsum, do, p']=tracking.dotdict(args=[s,'do','p'],kwargs=dict())

    return inputprofiles