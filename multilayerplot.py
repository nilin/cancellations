import matplotlib.pyplot as plt
import testing
import jax.numpy as jnp
import pickle


d=3
n=int(input('n? '))
samples=int(input('nsamples '))

def sq(x):
	return jnp.average(jnp.square(x))

yAR=[]
yAT=[]
yR=[]
yT=[]
#Ls=[1,2,4,8,16,32]
outs=dict()
Ls=jnp.arange(2,int(input('max number of layers ')))
for l in Ls:
	out=testing.test_multilayer(d=d,n=n,layers=l,samples=samples)
	outs[int(l)]=out
	AR,AT,R,T=(out[k] for k in ['AR','AT','R','T'])
	yAR.append(sq(AR)/sq(R))
	yAT.append(sq(AT)/sq(T))
	yR.append(sq(R))
	yT.append(sq(T))
	print(str(l)+100*'=')

fn='n='+str(n)+' '+'{:,}'.format(samples)+'samples'

with open('data multilayer '+fn,'wb') as f:
	pickle.dump(outs,f)

plt.plot(Ls,yAR,'b')
plt.plot(Ls,yAT,'r')
plt.plot(Ls,yR,'b:')
plt.plot(Ls,yT,'r:')

plt.yscale('log')
plt.savefig('plots multilayer '+fn+'.pdf')


