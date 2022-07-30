import e2
import pdb
import plottools as pt
import config as cfg
import AS_functions








if __name__=='__main__':


	exname='e2'
	for ac in ['tanh','ReLU']:

		prepath='outputs/{}/{}/'.format(exname,ac)
		path=cfg.longestduration('outputs/{}/{}/'.format(exname,ac))

		cfg.retrievestate(path+'/hist')
		emptylearner=AS_functions.gen_learner(*cfg.getval('learnerinitparams'))

		plotter=e2.Plotter([],[])
		plotter.clonefrom(path+'/hist')
		plotter.prep(emptylearner,cfg.expsched(.1,cfg.hour))

		cfg.outpaths={prepath}
		plotter.plot3()
		plotter.plotlosshist()
		plotter.plotweightnorm()
		
