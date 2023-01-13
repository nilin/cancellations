from cancellations.config import batchjob, browse, config as cfg, sysutil, tracking
from cancellations.utilities import textutil
import re, os, importlib, sys




class Run(batchjob.Batchjob):

    def execprocess(self):

        tasks=[\
            ('cancellations.examples.Barronnorm','Run'),\
            ('cancellations.examples.SI','Run'),\
            ('cancellations.examples.comparenorms','Genfns'),\
            ('cancellations.examples.comparenorms','Compare'),\
            ('cancellations.examples.game','Run'),\
            #'cancellations/run/unsupervised.py',\
            #'cancellations/plotting/plotmultiple.py'\
            ]

        mname,classname=self.run_subprocess(browse.Browse(options=tasks,displayoption=lambda o : o[0]+'.'+o[1]))
        m = importlib.import_module(mname)
        cls = getattr(m,classname)
        self.run=cls()
        self.run_subprocess(self.run,taskname='run script')


    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile().butwith(tasks=['pick script','pick profile','run script'],**kw)


def debug(disable_jit=True):
    from cancellations.config import sysutil
    import jax
    #sysutil.clearscreen()
    cfg.debug=True
    cfg.display_on=False
    if disable_jit:
        with jax.disable_jit(): Run().run_as_NODISPLAY()
    else:
        Run().run_as_NODISPLAY()

if __name__=='__main__':
    if 'd' in sys.argv:
        debug()
    elif 'd2' in sys.argv:
        debug(disable_jit=False)
    else:
        Run().run_as_main()