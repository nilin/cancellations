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

biasinitsize=.1
initweight_coefficient=2
layernormalization=None
plotfineness=50

def agrees(d1,**d2):
    return all([d1[k]==d2[k] for k in d1.keys() if k in d2.keys()])

def test():
    print(agrees(dict(a=1,b=2,c=3),b=1))
    print(agrees(dict(a=1,b=2,c=3),b=2))
