import gen_outputs
import plot_twolayer
import sys
import jax.numpy as jnp


if len(sys.argv)==1:
	print('\n\nquickgen_twolayerplot.py nmax scaling=X/H\n\n')
	quit()
nmax=int(sys.argv[1])
scaling=sys.argv[2]


acs=['DReLU_normalized','tanh']



for samples in 2**jnp.arange(1,10):
	for depth in [3,4,5]:

		instances=samples

		for n in range(2,nmax+1):

			for ac in acs:	
				gen_outputs.generate(n,n,depth,ac,scaling,instances,samples,'silent')

	
	print('\n\nyou now have data to run \nplot_twolayer.py '+str(nmax)+' '+scaling+' '+100*'='+'\n\n')

