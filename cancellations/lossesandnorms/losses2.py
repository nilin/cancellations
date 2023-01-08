import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_map
from . import losses

#dontpick



####################################################################################################
####################################################################################################
#
# loss that DZY proposed
#
####################################################################################################
####################################################################################################


class Lossgrad_separate_denominators(losses.Lossgrad_memory):

    def __init__(self,period,microbatchsize,g,rho,**kw):
        super().__init__(period,microbatchsize,g,rho,**kw)
        self.gradestimate=jax.jit(partial(self.gradestimate,self.g,self.Dg,self.rho))

    @staticmethod
    def gradestimate(g,Dg,rho,params,X1,X2,f1,f2,_Efg,_Eff,Egg):

        g1,g2=g(params,X1),g(params,X2)
        Dg2=Dg(params,X2)
        rho1,rho2=rho(X1),rho(X2)

        fg=jnp.average(f1*g1/rho1)

        gradweights1=-f2/rho2
        gradweights2=(fg/rho1)*g2/rho2

        assert(gradweights1.shape==rho2.shape)

        Z=jnp.sqrt(Egg)
        Z3=Egg*Z

        return tree_map(lambda D:\
            jnp.tensordot(gradweights1,D,axes=(0,0))/Z\
            +jnp.tensordot(gradweights2,D,axes=(0,0))/Z3,\
            Dg2)


####################################################################################################