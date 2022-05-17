import gen_outputs
import plot_twolayer
import sys


if sys.argv[1]=='h':
	print('\n\nquickgen_twolayerplot.py nmax scaling=X/H\n\n')
	quit()
nmax=int(sys.argv[1])
scaling=sys.argv[2]


acs=['ReLU','tanh','HS','exp']





def generate(nmax,scaling,instances,samples,plot=False):
	for n in range(2,nmax+1):
		for ac in acs:	
			if plot:
				plot_twolayer.makeplot(nmax,scaling)	
			if n>9 and ac=='exp':
				continue
			gen_outputs.generate(n,n,2,ac,scaling,instances,samples,'silent')




print('priming: small dataset for each n')
generate(nmax,scaling,2,1)
print('priming done')

instances=1
samples=1

while True:
	samples=samples*16
	instances=instances*4
	generate(nmax,scaling,instances,samples)

