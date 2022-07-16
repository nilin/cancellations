# modification to Lin's test

# Use a two layer network to fit a Gaussian function. 
#
# The implementation is via jax.

import GPU_sum
import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import bookkeep as bk
#from GPU_sum import sum_perms_multilayer as sumperms
import GPU_sum
import optax
import math
import universality
import sys
import matplotlib.pyplot as plt
#from plotuniversal import plot as plot3
import numpy as np
import time




class Trainer:
    def __init__(self,d,n,m,samples):
        self.d,self.n,self.m,self.samples=d,n,m,samples

        k0=rnd.PRNGKey(0)
        self.W,self.b=universality.genW(k0,n,d,m)

        [X_train, Z_train] = bk.get('data/train/gaussian_1')

        self.X_train=X_train[:samples]
        self.Z_train=Z_train[:samples]

        self.opt=optax.adamw(.01)
        self.state=self.opt.init((self.W,self.b))

        self.paramshistory=[]

        self.epochlosses=[]

    def epochNS(self,minibatchsize):

        #X_train,Z_train=randperm(self.X_train,self.Z_train)
        X_train,Z_train=self.X_train,self.Z_train

        losses=[]

        for a in range(0,self.samples,minibatchsize):
            c=min(a+minibatchsize,self.samples)

            X=X_train[a:c]
            Z=Z_train[a:c]

            grad,loss=universality.lossgradNS((self.W,self.b),X,Z)

            updates,self.state=self.opt.update(grad,self.state,(self.W,self.b))
            (self.W,self.b)=optax.apply_updates((self.W,self.b),updates)

            rloss=loss/universality.lossfnNS(Z,0)
            losses.append(rloss)
            bk.printbar(rloss,rloss)
        self.epochlosses.append(losses)


    def checkpoint(self):
        self.paramshistory.append((self.W,self.b))
        return jnp.average(self.epochlosses[-1])


    def savehist(self,filename):
        bk.save(self.paramshistory,filename)




def initandtrain(d,n,m,samples,batchsize,traintime,trainmode='AS'):
    T=Trainer(d,n,m,samples)

    variables={'d':d,'n':n,'m':m,'s':samples,'bs':batchsize}


    t0=time.perf_counter()
    loss='null'

    while time.perf_counter()<t0+traintime and (loss=='null' or loss>.0001):
        T.epochNS(batchsize)
        loss=T.checkpoint()
        T.savehist('data/hists/'+trainmode+'_'+formatvars_(variables))



def testerrorNS(Wb,samples=100):
    W,b=Wb
    m,n,d=W[0].shape
    # [X, Z] = bk.get('data/test/gaussian_1')
    [X, Z] = bk.get('data/train/gaussian_1')

    X=X[:samples]
    Z=Z[:samples]

    return universality.batchlossNS(Wb,X,Z)/universality.lossfn(Z,0)



def ploterrorhist(n,fn):
    hist=bk.get(fn)
    errorhist=[testerrorNS(Wb) for Wb in hist]
    error=errorhist[-1]

    plt.plot(errorhist,ls='dotted',label=str(n))

    plt.show() 

def formatvars_(D):
    D_={k:v for k,v in D.items() if k not in {'s','bs'}}
    return bk.formatvars_(D_)

def gen_Xs(key,d,n,samples):
	return rnd.normal(key,(samples,n,d))	

def regspaced_X(samples,r=2):
	X=jnp.arange(samples)*2*r/samples-r
	for _ in range(2):
		X=jnp.expand_dims(X,axis=-1)
	return X


def gaussian(X):
    return jnp.exp(-X**2)
    


if __name__=="__main__":

    if len(sys.argv)==1:
        print('\n\n --------------run with args: traintime----------------\n\n')

    samples=100
    batchsize=100

    d = 1
    n = 1

    bk.log('generating training data')
    
    #X = gen_Xs(rnd.PRNGKey(0),d,n,samples)*3
    X=regspaced_X(samples)
    Z = gaussian(X)
    bk.save(jnp.stack([X,Z]),'data/train/gaussian_'+str(d))

    bk.log('generating testing data')

    #X = gen_Xs(rnd.PRNGKey(1),d,n,samples)
    X=regspaced_X(samples)
    Z = gaussian(X)
    bk.save(jnp.stack([X,Z]),'data/test/gaussian_'+str(d))
    
    bk.log('training')

    traintime=float(sys.argv[1])
    trainmode='NS'

    m=1000
    initandtrain(d,n,m,samples,batchsize,traintime,trainmode)
    
    bk.log('plotting')

    fn='data/hists/'+trainmode+'_'+bk.formatvars_({'d':d,'n':n,'m':m})
    
    hist=bk.get(fn)
    W,b=hist[-1]
    [X_train, Z_train] = bk.get('data/train/gaussian_1')
    Z_nn = universality.nonsym(W,b,X)
    #plt.plot(jnp.squeeze(X),jnp.squeeze(Z),'bo',
    #        jnp.squeeze(X),jnp.squeeze(Z_nn),'rd',markersize=1)
    plt.plot(jnp.squeeze(X),jnp.squeeze(Z),'b',
            jnp.squeeze(X),jnp.squeeze(Z_nn),'r')
    plt.show()
 
    
    # ploterrorhist(n,fn)
