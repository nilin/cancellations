from cancellations.config import batchjob, browse, config as cfg, sysutil, tracking
from cancellations.utilities import textutil
import re, os, importlib, sys




class Run(batchjob.Batchjob):

    def execprocess(self):

        tasks=[\
            ('cancellations.examples.Barronnorm','Run'),\
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

        if cfg.debug:
            cfg.postprocesses.append(self.run)
            cfg.display_on=False
            return

        self.run_subprocess(self.run,taskname='run script')


    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile().butwith(tasks=['pick script','pick profile','run script'],**kw)


def main():
    Run().run_as_main()
    cfg.run_afterdisplayclosed()

def debug(disable_jit=True):
    from cancellations.config import sysutil
    import jax
    sysutil.clearscreen()
    cfg.debug=True
    if disable_jit:
        with jax.disable_jit():
            main()
    else: main()

if __name__=='__main__':
    if 'd' in sys.argv:
        debug()
    elif 'd2' in sys.argv:
        debug(disable_jit=False)
    else:
        main()