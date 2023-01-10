# import run
# from cancellations.utilities import sysutil
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


from cancellations.functions import _functions_, symmetries
import jax.random as rnd
import jax.numpy as jnp
from numpy.testing import assert_allclose

n=5; d=3
m=100

fdescr1=_functions_.ASBarron(n=n,d=d,m=m)
fdescr2=_functions_.ASNN(activation='relu',n=n,d=d,widths=[n*d,m,1])

(W,b)=fdescr1.weights
W2=jnp.reshape(W,(m,n*d))
#W=jnp.reshape(W,(-1,n,d))
params2=[(W2,b),(jnp.ones((1,m)),jnp.ones((1,)))]


X=rnd.normal(rnd.PRNGKey(0),(1000,n,d))
y1=jnp.squeeze(fdescr1.eval(X))
y2=jnp.squeeze(fdescr2._eval_(params2,X))

assert_allclose(y1,y2)

print('test passed')

# print()
# print()
# print()
# print()
# print(W)
# print(bs)
# print(a)
# print()
# print()
# print()
# print()

# f3=symmetries.gen_singlelayer_Af(n,d,'relu')
# 
# X=rnd.normal(rnd.PRNGKey(0),(1000,n,d))
# y1=jnp.squeeze(fdescr1._eval_(params,X))
# y2=jnp.squeeze(fdescr2._eval_(params,X))
# y3=jnp.squeeze(f3([(W,bs),a],X))
# 
# #breakpoint()
# print(y1.shape)
# print(y2.shape)
# print(y3.shape)
# assert_allclose(y1,y3)
# assert_allclose(y1,y2)
# print('test passed')