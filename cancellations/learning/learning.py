#
# nilin
#
# 2022/7
#



import optax


#----------------------------------------------------------------------------------------------------
# training object
#----------------------------------------------------------------------------------------------------

    
class Trainer():
    def __init__(self,lossgrad,learner,sampler,learning_rate=.01,**kwargs):

        self.lossgrad=lossgrad
        self.learner=learner
        self.sampler=sampler

        #self.samples,self.n,self.d=X.shape
        self.opt=optax.adamw(learning_rate,**{k:val for k,val in kwargs.items() if k in ['weight_decay','mask']})
        self.state=self.opt.init(self.learner.weights)
        

    def minibatch_step(self,X_mini,*Y_mini):
        loss,grad=self.lossgrad(self.learner.weights,X_mini,*Y_mini)
        updates,self.state=self.opt.update(grad,self.state,self.learner.weights)
        self.learner.weights=optax.apply_updates(self.learner.weights,updates)
        return loss


    def step(self):
        (X_mini,*Y_mini)=self.sampler.step()
        return self.minibatch_step(X_mini,*Y_mini)    



