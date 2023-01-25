import jax, sys
if '64' in sys.argv: jax.config.update("jax_enable_x64", True)

from cancellations.tracking import browse, runconfig as cfg, tracking, sysutil
import importlib, os



class Run(browse.Process):

    def execprocess(self):

        tasks=[\
            #('cancellations.examples.Barronnorm','Run'),\
            #('cancellations.examples.Barronnorm','Runthrough'),\
            #('cancellations.examples.Barronnorm','Plot'),\
            #('cancellations.examples.expslater','Run'),\
            ('cancellations.examples.experiments','Runthrough'),\
            ('cancellations.examples.SI','Run'),\
            ('cancellations.examples.SI','Plot'),\
            #('cancellations.examples.comparenorms','Genfns'),\
            #('cancellations.examples.comparenorms','Compare'),\
            #('cancellations.examples.game','Run'),\
            #'cancellations/run/unsupervised.py',\
            #'cancellations/plotting/plotmultiple.py'\
            ]

        mname,classname=tracking.runprocess(browse.Browse(options=tasks,displayoption=lambda o : o[0]+'.'+o[1]))
        m = importlib.import_module(mname)
        cls = getattr(m,classname)
        self.run=cls(profile=cls.getprofile(self))
        tracking.runprocess(self.run)


if __name__=='__main__':

    cfg.istest=('t' in sys.argv)
    cfg.debug=('d' in sys.argv)
    cfg.display_on=('n' not in sys.argv)
    cfg.dump=('dump' in sys.argv or 'D' in sys.argv)

    if cfg.debug:
        import jax
        cfg.display_on=False
        with jax.disable_jit(): Run().run_as_NODISPLAY()

    elif not cfg.display_on:
        cfg.display_on=False
        Run().run_as_NODISPLAY()

    else:
        Run().run_as_main()