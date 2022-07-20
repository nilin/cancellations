# Fit product of the first 4 eigenstates of a harmonic oscillator
#
# Antisymmetrized data and antisymmetrized network.


import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
import optax
import math
import AS_tools
#import universality
import sys
import matplotlib.pyplot as plt
#from plotuniversal import plot as plot3
import numpy as np
import time
import permutations_simple as ps
import targets
import train

fn_ex='gaussian_9'





def testerror(Wb,samples=100):
    W,b=Wb
    m,n,d=W[0].shape
    [X, Z] = bk.get('data/test/'+fn_ex)

    X=X[:samples]
    Z=Z[:samples]

    return train.batchlossAS(Wb,X,Z)/train.lossfn(Z,0)



def ploterrorhist(n,fn):
    samples_test = 200
    hist=bk.get(fn)
    errorhist=[testerror(Wb, samples_test) for Wb in hist]
    error=errorhist[-1]

    plt.plot(errorhist,ls='dotted',label=str(n))
    plt.show() 

def formatvars_(D):
    D_={k:v for k,v in D.items() if k not in {'s','bs'}}
    return bk.formatvars_(D_)

def gen_Xs(key,d,n,samples):
    X_r = rnd.normal(key,(samples,n,d))	
    # add soft bound constraint so the data is in [-1,1]
    X = jax.scipy.special.erf(X_r)
    return X







if __name__=="__main__":

    n=5
    samples=10000
    batchsize=100
    depth=3
    width=100

    bk.getparams(globals(),sys.argv)

    d = 1
    target_func=targets.HermiteSlater(n,'He',1/8)

    bk.log('generating training data')
    
    X = gen_Xs(rnd.PRNGKey(0),d,n,samples)

    Z = target_func(X)

    Z_std  = jnp.std(Z)
    print('std(Z)     = ', Z_std)

    Z = Z / Z_std

    bk.save([X,Z],'data/train/'+fn_ex)

    bk.log('generating testing data')

    X = gen_Xs(rnd.PRNGKey(1),d,n,1000)
    Z = target_func(X)

    # normalize according to the input data for consistency
    Z = Z / Z_std

    bk.save([X,Z],'data/test/'+fn_ex)
    
    bk.log('training')

    traintime=3000
    trainmode='AS'

    m=100
    train.initandtrain('data/train/'+fn_ex,'data/hists/'+fn_ex,trainmode,[width]*(depth-1),samples,batchsize,stopwhenstale=False)
    
    bk.log('plotting')

    hist=bk.get('data/hists/'+fn_ex)
    W,b=hist[-1]

    # overall test
    [X_test, Z_test] = bk.get('data/test/'+fn_ex)


    samples_plot = np.min([1000, samples])
    X = X_test.copy()[:samples_plot,:,:]
    for l in range(5):
        x_slice = jax.scipy.special.erf(np.random.randn(n)) * 0.5
        print("x_slice = ", x_slice[:-1])
        for j in range(n-1):
            X[:,j,:] = x_slice[j]
        Z = target_func(X)
        Z = Z/Z_std
        Z_nn = AS_tools.AS_NN(W,b,X)
        plt.plot(jnp.squeeze(X[:,n-1,:]),jnp.squeeze(Z),'bo',
                jnp.squeeze(X[:,n-1,:]),jnp.squeeze(Z_nn),'rd')
    
        plt.show()

    ploterrorhist(n,'data/hists/'+fn_ex)
