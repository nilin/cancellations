
import sys
import jax

if not '32' in sys.argv:
    jax.config.update("jax_enable_x64", True)



display_on=True
debug=False



class noRun:
    def run_as_NODISPLAY(self): pass

postcommand=lambda: None
postrun=noRun()

