import e2
import pdb
import plottools as pt
import config as cfg
import AS_functions






def prep_plotter(histpath,timestep,timebound):
	plotter=e2.Plotter([],[])
	cfg.retrievestate(histpath)
	plotter.clonefrom(histpath)
	plotter.filtersnapshots(cfg.periodicsched(timestep,timebound))
	plotter.loadlearnerclone()
	return plotter




if __name__=='__main__':


	timestep1=5
	timestep2=5
	timebound=300


	exname=cfg.cmdparams[0]
	print(exname)

	for k,val in cfg.cmdredefs.items():
		locals()[k]=int(val)
		print('{}={}'.format(k,val))



	for ac in ['tanh','ReLU']:

		prepath='outputs/{}/{}/'.format(exname,ac)
		path=cfg.longestduration('outputs/{}/{}/'.format(exname,ac))
		histpath=path+'/hist'
		outpath=prepath+'processed '+cfg.nowstr()+'/'
		cfg.outpaths={outpath}

		print('output will be saved to\n'+outpath+'\n')


		plotter=prep_plotter(histpath,timestep1,timebound)
		#----------------------------------------------------------------------------------------------------	
		plotter.prep()
		plotter.plot3()
		plotter.plotlosshist()
		plotter.plotweightnorms()
		#----------------------------------------------------------------------------------------------------	

		
		plotter=prep_plotter(histpath,timestep2,timebound)
		for t,snapshot in zip(*plotter.gethist('weights')):
			print('function plot for time {}'.format(int(t)))
			plotter.plotfn(plotter.getstaticlearner(snapshot),figname='fnplots/{} s'.format(int(t)))

		
