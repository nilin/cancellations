import display as disp
import config as cfg
import curses as cs



class ConcreteDisplay(disp.StackedDisplay):
	def __init__(self,xlim,ylim,**kw):
		super().__init__(**kw)
		self.xlim=xlim
		self.ylim=ylim
		x0,x1=self.xlim
		y0,y1=self.ylim
		self.width=x1-x0
		self.height=y1-y0
		self.pad=cs.newpad(self.height,self.width)	


	def draw(self):
		self.pad.erase()
		x1,x2=self.xlim
		y1,y2=self.ylim
		lines=self.getlines()
		for i,line in enumerate(lines):
			self.pad.addstr(i,0,line)

		self.pad.refresh(0,0,y1,x1,y2,x2)

	def poke(self,src):
		self.draw()

def getinput(*args,**kw):
	a=getscreen().getch()
	cs.flushinp()
	cfg.currentprofile().dashboard.draw()
	return cfg.extractkey_cs(a)

def getscreen(): return cfg.screen
cfg.getinput=getinput

def run_in_display(runfn,profile,*a,**kw):
	cfg._currentprofile_=profile
	def wrapped(screen):
		cfg.screen=screen
		screen.nodelay(True)
		cs.use_default_colors()
		profile.dashboard=disp.CDashboard(width=cs.COLS,height=cs.LINES)
		try: profile.prepdashboard()
		except: pass
		runfn(*a,**kw)
	cs.wrapper(wrapped)

def clear():
	cfg.screen.clear()
	cfg.screen.refresh()

def subtask_in_display(runfn,profile,*args,**kw):
	profile.dashboard=disp.CDashboard(width=cs.COLS,height=cs.LINES)
	outerprofile=cfg.currentprofile()
	cfg._currentprofile_=profile
	clear()
	out=runfn(*args,**kw)
	clear()
	cfg._currentprofile_=outerprofile
	return out


# test
#
#if __name__=='__main__':
#
#	import time
#
#	def wrapped(screen):
#		#cs.use_default_colors()
#		h=cs.LINES
#		w=cs.COLS
#
#		dashboard=CDashboard(50,20)
#
#		S=db.StackedDisplay(memory=session,width=w)
#		C=dashboard.add(S,(0,w-1),(0,10))
#		S.add(db.StaticText('test\n1\n2'))
#		S.add(db.Bar('y'))
#
#		dashboard.draw_all()
#
#		screen.nodelay(True)
#
#		for Y in range(100):
#
#			c=screen.getch()
#
#			screen.addstr(20,0,str(c))
#
#			session.remember('y',Y/100)
#			C.draw()
#			time.sleep(1)
#
#
#	cs.wrapper(wrapped)