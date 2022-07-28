import sys
import config as cfg
import e1,e2



class LogListener():
	def poke(*y,**z):
		print(cfg.getval('log'))	




cfg.addlistener(LogListener(),'log')



e={'e1':e1,'e2':e2}[sys.argv[1]]
e.run(sys.argv[2:])


