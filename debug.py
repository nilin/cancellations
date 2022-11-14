from cancellations.utilities import setup, tracking, browse
import jax
from cancellations.utilities import sysutil

sysutil.clearscreen()

def debug():
    with jax.disable_jit():
        import run
        run.main()

if __name__=='__main__':
    setup.debug=True
    debug()