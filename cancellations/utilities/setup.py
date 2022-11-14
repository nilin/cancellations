
import sys
import jax

if not '32' in sys.argv:
    jax.config.update("jax_enable_x64", True)



display_on=True
debug=False

postcommands=[]
postprocesses=[]


defaultinputs=dict()


def run_afterdisplayclosed():
    globals()['display_on']=False

    for postrun in postprocesses:
        postrun.run_as_NODISPLAY()

    for postcommand in postcommands:
        postcommand()