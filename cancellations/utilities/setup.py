
import sys
import jax

if not '32' in sys.argv:
    jax.config.update("jax_enable_x64", True)


postrun=lambda: None
display_on=True
debug=False

