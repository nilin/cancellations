import jax
import config as cfg
import importlib
import sys

def _print_(*args,**kw):
	with jax.disable_jit():
		print(*args,**kw)


cfg.print=_print_
cfg.displaymode='logdisplay'



importlib.import_module(sys.argv[1])
