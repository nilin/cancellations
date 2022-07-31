import e1,e2
import pdb
import plottools as pt
import config as cfg
import AS_functions






def load_plotter(histpath,schedule):
	plotter=e1.LoadedPlotter(histpath)
	#plotter.clonefrom(histpath)
	#plotter.loadlearnerclone()
	#cfg.retrievestate(histpath)
	plotter.filtersnapshots(schedule)
	return plotter


activations=['ReLU','tanh']

if __name__=='__main__':


	timebound=3600


	exname=cfg.cmdparams[0]
	print(exname)

	for k,val in cfg.cmdredefs.items():
		locals()[k]=int(val)
		print('{}={}'.format(k,val))



	for ac in ['tanh','ReLU']:

		path='outputs/{}/{}/'.format(exname,ac)
		longest_run_path=cfg.longestduration(path)
		histpath=longest_run_path+'/hist'
		outpath=path+'processed '+cfg.nowstr()+'/'
		cfg.outpaths={outpath}

		print('output will be saved to\n'+outpath+'\n')


		#----------------------------------------------------------------------------------------------------	
		plotter=load_plotter(histpath,cfg.periodicsched(5,timebound))
		plotter.prep()
		plotter.plot3()
		
		#----------------------------------------------------------------------------------------------------	
		plotter=load_plotter(histpath,cfg.stepwiseperiodicsched([2,10,60],[0,60,120,timebound]))
		plotter.prep()
		plotter.plotweightnorms()
		plotter.plotlosshist()

		
		#----------------------------------------------------------------------------------------------------	
		plotter=load_plotter(histpath,cfg.expsched(1,timebound,2))
		for t,snapshot in zip(*plotter.gethist('weights')):
			print('function plot for time {}'.format(int(t)))
			plotter.plotfn(plotter.getstaticlearner(snapshot),figname='fnplots/{} s'.format(int(t)))



			
	cp=e2.CompPlotter({ac:cfg.longestduration('outputs/{}/{}/'.format(exname,ac))+'/hist' for ac in activations})
	cp.prep(cfg.expsched(1,timebound,.2))
	cfg.outpaths={'outputs/{}/comparison/processed {}/'.format(exname,cfg.nowstr())}
	cp.compareweightnorms()
	cp.comp3()




