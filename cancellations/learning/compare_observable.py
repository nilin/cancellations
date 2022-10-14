import browse
import mcmc
import config as cfg
import os
import jax.random as rnd
import jax.numpy as jnp
import cdisplay
import jax
import display
from example import profile as exprofile



profile=cfg.Profile(
    proposalfn=lambda key,X: X+rnd.normal(key,X.shape)/25,
    Observable=jax.jit(lambda X: jnp.sum(X**2,axis=(-2,-1))),\
    nrunners=1000\
)
cfg._currentprofile_=profile






profile.act_on_input=cfg.donothing
cfg.checkforinput=cfg.donothing

path,_=browse.pickfolders_leave_cs(msg='test',condition=lambda path:os.path.exists(path+'/data/setup'))



display.clear() 
setupdata=cfg.Profile(**cfg.load(path+'/data/setup'))

target=setupdata.target.restore()
learner=setupdata.learner.restore()
_,n,d=setupdata.X_train.shape
X0_distr=lambda key,samples:exprofile._X_distr_(key,samples,n,d)
envelope=exprofile.envelope

p0=jax.jit(lambda X:envelope(X)*target.eval(X)**2)
p=jax.jit(lambda X:envelope(X)*learner.eval(X)**2)

sampler=mcmc.Sampler(p0,profile.proposalfn,X0_distr(tracking.nextkey()(),profile.nrunners),burnsteps=0)

@jax.jit
def Ep(X):
    ratio=p(X)/p0(X)
    return jnp.sum(ratio*O(X))/jnp.sum(ratio)


for i in range(100):
    sampler.step()
    print('MCMC burning step {}'.format(i),end='\r')
print()

E0_estimates=[]
E1_estimates=[]

O=profile.Observable

for i in range(100000):
    sampler.step()
    X=sampler.X

    if i%10==0:
        E0_estimates.append(jnp.sum(O(X))/X.shape[0])
        E1_estimates.append(Ep(X))

    print('MCMC estimates [O]_p0~{:.4f} vs [O]_p~{:.4f}, {:2.2%}'.format(
        E0_estimates[-1],E1_estimates[-1],jnp.log(E0_estimates[-1]/E1_estimates[-1])),end='\r')


