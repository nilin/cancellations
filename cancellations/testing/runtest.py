# import run
# from cancellations.utilities import sysutil, setup
# import jax
# 
# sysutil.clearscreen()
# setup.debug=True
# with jax.disable_jit():
#     run.main()

#from cancellations.utilities import permutations
#import jax.random as rnd
#import jax.numpy as jnp
#
#s=1000
#n=4
#Ps,signs=permutations.allpermtuples(n)
#M=jnp.reshape(jnp.arange(n**2),(n,n))
#M2=jnp.ones((100))[:,None,None]*M[None,:,:]
#I=(jnp.ones((Ps.shape[0]))[:,None]*jnp.arange(n)[None,:]).astype(int)


from cancellations.functions import functions, NNfunctions
from jax.nn import relu
import jax.random as rnd
import jax.numpy as jnp
from numpy.testing import assert_allclose

n=5; d=3;
m=100

fdescr=functions.ASNN(activation='relu',n=n,d=d,widths=[n*d,m,1])
params=fdescr.weights
[(W,bs),(a,_)]=params
W=jnp.reshape(W,(-1,n,d))

f2=NNfunctions.gen_singlelayer_Af(n,relu)

X=rnd.normal(rnd.PRNGKey(0),(1000,n,d))
y1=f1=fdescr.eval(X)
y2=f2([(W,bs),a],X)

assert_allclose(y1,y2)
print('test passed')