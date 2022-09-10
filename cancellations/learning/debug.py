import jax
import config as cfg
import sys

def print(*args,**kw):
	with jax.disable_jit():
		print(*args,**kw)


cfg.print=_print_
cfg.displaymode='logdisplay'


if __name__=='__main__':
	import importlib
	importlib.import_module(sys.argv[1])
