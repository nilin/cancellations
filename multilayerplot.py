import matplotlib.pyplot as plt
import testing
import jax.numpy as jnp
import pickle


d=3
n=int(input('n? '))
samples=int(input('nsamples '))

def sq(x):
	return jnp.average(jnp.square(x))

def plotdict(X,Y,*args):
	plt.plot(X,[Y[int(x)] for x in X],*args)


antisymmetrized={ac:dict() for ac in testing.acs}
nonsymmetrized={ac:dict() for ac in testing.acs}

Ls=jnp.arange(2,int(input('max number of layers ')))
for l in Ls:
	antisymmetrized_,nonsymmetrized_=testing.test_multilayer(d=d,n=n,layers=l,samples=samples)
	for ac in testing.acs:
		antisymmetrized[ac][int(l)]=sq(antisymmetrized_[ac])/sq(nonsymmetrized_[ac])
		nonsymmetrized[ac][int(l)]=sq(nonsymmetrized_[ac])

	print(str(l)+100*'=')

fn='n='+str(n)+' '+'{:,}'.format(samples)+'samples'

with open('data multilayer '+fn,'wb') as f:
	pickle.dump({'a':antisymmetrized,'n':nonsymmetrized},f)

plotdict(Ls,antisymmetrized['tanh'],'r')
plotdict(Ls,antisymmetrized['DReLU'],'g')
plotdict(Ls,antisymmetrized['ReLU'],'b')
plotdict(Ls,nonsymmetrized['tanh'],'r:')
plotdict(Ls,nonsymmetrized['DReLU'],'g:')
plotdict(Ls,nonsymmetrized['ReLU'],'b:')

plt.yscale('log')
plt.savefig('plots multilayer '+fn+'.pdf')


