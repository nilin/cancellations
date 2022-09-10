import display as disp
import config as cfg
import curses as cs



class ConcreteDisplay(disp.StackedDisplay):
	def __init__(self,*a,**kw):
		super().__init__(*a,**kw)
		self.pad=cs.newpad(self.height+1,self.width+1)	


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



class Pad(disp.CompositeDisplay):
	def __init__(self,xlim,ylim,WIDTH=None,HEIGHT=None):
		super().__init__(xlim,ylim)
		self.pad=cs.newpad(self.height if HEIGHT==None else HEIGHT,self.width if WIDTH==None else WIDTH)

	def refresh(self,y,x=0):
		x0,x1=self.xlim
		y0,y1=self.ylim
		self.pad.refresh(y,x,y0,x0,y1-1,x1-1)

	def draw(self):
		self.refresh(0,0)

	def addstr(self,*a,**kw): self.pad.addstr(*a,**kw)

	def erase(self): self.pad.erase(); self.pad.clear()



#	def draw(self):
#		cfg.screen.clear()
#		cfg.screen.refresh()
#		super().draw()
#		cfg.screen.refresh()



def checkforinput(*args,**kw):
	a=getscreen().getch()
	cs.flushinp()
	cfg.currentprocess().display.draw()
	return cfg.extractkey_cs(a)

cfg.checkforinput=checkforinput

def getscreen(): return cfg.screen

def clearscreen():
	getscreen().clear()
	getscreen().refresh()

def session_in_display(processfn,profile,nodelay=True,**kw):

	def wrapped(screen):
		cfg.screen=screen
		screen.nodelay(nodelay)
		cs.use_default_colors()
		cfg.session.display=disp.CompositeDisplay((0,cs.COLS),(0,cs.LINES))
		#try: profile.prepdashboard(profile.dashboard)
		#except: pass
		processfn(profile,display=cfg.session.display,**kw)

	out=cs.wrapper(wrapped)
	return out



def runtask(task,profile,display):
	cfg.loadprocess(task)
	output=task(profile,display)
	cfg.unloadprocess(task)
	clearscreen()
	return output


#def run_in_subdisplay(process,processfn,display,*args,**kw):
#
#	profile.dashboard=disp.CDashboard(width=cs.COLS,height=cs.LINES)
#	cfg._currentprofile_=profile
#	clear()
#	out=runfn(profile,*args,**kw)
#	clear()
#
#	return out
#
#
#def clear():
#	cfg.screen.clear()
#	cfg.screen.refresh()




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