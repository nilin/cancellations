import math
import jax.numpy as jnp
import matplotlib.pyplot as plt

a_HS=[math.pi/4,1/2,0,1/12,0,3/80,0,5/224,0,35/2304,0,63/5632,0,231/26624,0,143/20480,0,6435/1114112]
a_ReLU=(1/math.pi)*jnp.array([\
1/2,math.pi/4,1/4,0,1/48,0,1/160,0,5/1792,0,7/4608,0,21/22528,0,33/53248,0,143/327680\
])

N=min(len(a_HS),len(list(a_ReLU)))
even=range(0,N,2)
odd=range(1,N,2)

plt.yscale('log')
plt.scatter(range(len(a_HS)),a_HS,color='r')
plt.plot(odd,[a_HS[n] for n in odd],color='r')
plt.scatter(range(len(a_ReLU)),a_ReLU,color='b')
plt.plot(even,[a_ReLU[n] for n in even],color='b')
plt.show()
