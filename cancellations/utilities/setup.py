
import sys
import jax

if not '32' in sys.argv:
    jax.config.update("jax_enable_x64", True)



display_on=True
debug=False


#def testjitdisabled():
#    a=jax.numpy.ones((5,))
#    @jax.jit
#    def f(x):
#        print(x)
#        return 2*x
#    f(a)
#    f(a)
#    f(a)

if 'db' in sys.argv:
    debug=True
    #jax.disable_jit()
    #print('debugging mode: jit disabled')
    #testjitdisabled()

########


########

postcommands=[]
postprocesses=[]


defaultinputs=dict()


def run_afterdisplayclosed():
    globals()['display_on']=False

    for postrun in postprocesses:
        postrun.run_as_NODISPLAY()

    for postcommand in postcommands:
        postcommand()