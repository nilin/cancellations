from jax.config import config
config.update("jax_enable_x64", True)
import gen_outputs
import jax.numpy as jnp
import plot_twolayer
import sys


if len(sys.argv)==1:
	print('\n\nquickgen_twolayerplot.py nmax X/H outfolder\n\n')
	quit()
nmax=int(sys.argv[1])
scaling=sys.argv[2]
folder=sys.argv[3]
	


acs=['ReLU','tanh','HS','exp']



for samples in 10**jnp.arange(1,6):
	for n in range(2,nmax+1):
		for ac in acs:	
			if n>9 and ac=='exp':
				continue
			gen_outputs.generate_zip(n,n,2,ac,scaling,samples,'silent',folder=folder)

	
	print('\n\nyou now have data to run \nplot_twolayer.py '+str(nmax)+' '+scaling+' '+100*'='+'\n\n')

