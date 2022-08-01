import e1,e2
import pdb
import plottools as pt
import config as cfg
import jax.numpy as jnp
import AS_functions








activations=['ReLU','tanh']

if __name__=='__main__':


	timebound=3600


	exname=cfg.cmdparams[0]
	ex={'e1':e1,'e2':e2}[exname]

	print(exname)

	for k,val in cfg.cmdredefs.items():
		locals()[k]=int(val)
		print('{}={}'.format(k,val))



	for ac in ['tanh','ReLU']:


		try:		
			top_path='outputs/{}/{}/'.format(exname,ac)
			folders={'longest duration':cfg.longestduration(top_path),'newest':cfg.latest(top_path)}
			alreadyprocessed=set()

			for prop,folder in folders.items():

				print('\n'+prop)
				if folder in alreadyprocessed:
					print('already processed')
					continue

				alreadyprocessed.add(folder)	


				histpath=folder+'/hist'
				outpath=top_path+'processed '+cfg.nowstr()+'/'
				cfg.outpaths={outpath}

				print('output will be saved to\n'+outpath+'\n')


				#----------------------------------------------------------------------------------------------------	
				#plotter=pt.LoadedPlotter(histpath)
				#plotter.prep(cfg.periodicsched(5,timebound))
				#plotter.plot3()
				
				#----------------------------------------------------------------------------------------------------	
				plotter=pt.LoadedPlotter(histpath)
				#plotter.prep(cfg.stepwiseperiodicsched([5,10,60],[0,60,120,timebound]))
				plotter.prep(ex.learningplotsched)
				plotter.plotweightnorms()
				plotter.plotlosshist()

				
				#----------------------------------------------------------------------------------------------------	
				plotter=pt.LoadedPlotter(histpath)
				#plotter.filtersnapshots(cfg.expsched(5,timebound,1))
				plotter.filtersnapshots(ex.fnplotsched)
				for t,snapshot in zip(*plotter.gethist('weights')):
					print('function plot for time {}'.format(int(t)))
					plotter.plotfn(plotter.getstaticlearner(snapshot),figname='fnplots/{} s'.format(int(t)))

		except Exception as e:
			print('something went wrong, maybe data was not generated yet')
			print(str(e))



			
	try:		
		cp=pt.CompPlotter({ac:cfg.longestduration('outputs/{}/{}/'.format(exname,ac))+'/hist' for ac in activations})
		cp.prep(cfg.expsched(1,timebound,.2))
		cfg.outpaths={'outputs/{}/comparison/processed {}/'.format(exname,cfg.nowstr())}
		cp.compareweightnorms()
		#cp.plot3()

	except:
		print('something went wrong with comparison plot, maybe data was not generated for both activations yet')



