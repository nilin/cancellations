from cancellations.config import config
import jax
with jax.disable_jit():
    from cancellations.testing import runtest