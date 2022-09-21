from cancellations.utilities import setup, textutil
from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d,estimateobservables,profiles as P
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay, _display_
from cancellations.utilities import numutil, tracking, browse, batchjob, energy, sysutil
import jax.numpy as jnp
import jax
import os





class Run(_display_.Process):
    def execprocess(self):

        for i in range(5):
            self.log(str(i))

        T,B=self.display.vsplit(r=.7)
        L,R=T.hsplit()

        f=L.add(0,0,_display_._LogDisplay_(self,50,20))

        for i in range(0,100,10):
            R.add(i,i,_display_._TextDisplay_(textutil.lipsum))
        #self.display.add(0,30,_display_._TextDisplay_(textutil.lipsum))
        self.display.arm()
        self.display.drawall()
        T.getelementstrings()

        import time 
        for i in range(100):
            self.log(str(i))
            R.getcorner=lambda _self_: (i,i)
            self.display.drawall()

            time.sleep(.02)
#        sysutil.save(profile.p_descr.compress(),self.outpath+'density')
#        sysutil.save(profile.p_descr.compress(),self.outpath+'data/functions/density')
#        sysutil.save(profile.psi_descr.compress(),self.outpath+'data/functions/psi')
#        super().execprocess()


if __name__=='__main__':
    Run().run_as_main()
