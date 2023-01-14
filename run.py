from cancellations.config import browse, config as cfg, sysutil, tracking
from cancellations.display import _display_
from cancellations.utilities import textutil
import re, os, importlib, sys




class Run(_display_.Process):

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

        mname,classname=tracking.runprocess(browse.Browse(options=tasks,displayoption=lambda o : o[0]+'.'+o[1]))
        m = importlib.import_module(mname)
        cls = getattr(m,classname)
        self.run=cls()
        tracking.runprocess(self.run)


if __name__=='__main__':

    if 'd' in sys.argv:
        import jax
        cfg.debug=True
        cfg.display_on=False
        with jax.disable_jit(): Run().run_as_NODISPLAY()

    elif 'n' in sys.argv:
        cfg.display_on=False
        Run().run_as_NODISPLAY()

    else:
        Run().run_as_main()