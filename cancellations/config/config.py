import sys
import jax

if not '32' in sys.argv:
    jax.config.update("jax_enable_x64", True)

display_on=True
debug=False

biasinitsize=.1
initweight_coefficient=2
