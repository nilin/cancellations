import jax
import math
import util
import jax.numpy as jnp

def gen_att(omega):
    def att(Q,K,V):
        sim=omega(jax.vmap(jnp.inner,in_axes=(0,0))(Q,K))
        return jax.vmap(jnp.dot,in_axes=(0,0))(sim,V)
    return jax.jit(att)

def gen_multihead(omega):
    att=gen_att(omega)

    def multihead(Ws,Q,K,V):
        Wqs,Wks,Wvs,WO=Ws
        Q_=jnp.inner(Q,Wqs)
        K_=jnp.inner(K,Wks)
        V_=jnp.inner(V,Wvs)
        concatenation=jax.vmap(att,in_axes=(-2,-2,-2),out_axes=-1)(Q_,K_,V_)
        return jnp.inner(util.collapselast(concatenation,2),WO)
    return jax.jit(multihead)

def gen_simple_SAB(omega):
    mh=gen_multihead(omega)
    return jax.jit(lambda Ws,Y:mh(Ws,Y,Y,Y))

#	phi_iJ=jax.vmap(phi,in_axes=(None,None,-2),out_axes=-1)
#	def pooled_along_y(params,xi,yJ):
#		return pool(phi_iJ(params,xi,yJ),axis=-1)
#
#	return jax.jit(jax.vmap(pooled_along_y,in_axes=(None,-2,None),out_axes=-2))

def initweights_SimpleSAB(h,d,*args,**kw):
    dq_,dv_=math.ceil(d/h),math.ceil(d/h)
    Wqs=util.initweights((h,dq_,d))
    Wks=util.initweights((h,dq_,d))
    Wvs=util.initweights((h,dv_,d))
    WO=util.initweights((d,h*dv_))
    return Wqs,Wks,Wvs,WO