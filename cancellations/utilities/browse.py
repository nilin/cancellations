import os
import re
import curses as cs
import pdb

from . import tracking
from ..display.display import StaticText, dash
from ..display import display as disp
from ..display import cdisplay
from ..utilities import config as cfg,sysutil,tracking
import time

from cancellations import utilities


up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'




def defaultpathprofile():
	parentfolder='outputs/'
	return tracking.Profile(name='filterpaths',\
	parentfolder=parentfolder,\
	regex='.*',\
	condition1=lambda path:os.path.exists(parentfolder+path+'/data/setup'),\
	dynamiccondition=None,\
	)


class Browse(cdisplay.Process):
	def execprocess(process):
		profile,display=process,process.display
		#browsing=tracking.Process(profile,display=display)

		W=display.width
		H=display.height
		x0,x3=display.xlim
		y0,y1=display.ylim
		x1,x2=round(x0*.75+x3*.25), round(.4*x0+.6*x3)
		screen=cfg.screen
		screen.nodelay(False)

		explanation=profile.msg
		#explainpad.addstr(0,0,explanation)
		#explainpad.draw()


		#explainpad=cdisplay.Pad((x0,x1-5),(y0,y1))
		explanationtextdisp,_=display.add(cdisplay.ConcreteStaticTextDisplay((x0,x1-5),(y0,y1),msg=explanation),name='explanation')
		#explanationtextdisp.draw()

		listpad=cdisplay.Pad((x1,x2-10),(y0,y1),100,1000)
		#matchinfopad=cdisplay.Pad((x2,x3),(y0,y1))
		matchinfotextdisp,_=display.add(cdisplay.ConcreteStaticTextDisplay((x2,x3),(y0,y1),msg=''),name='matchinfo')

		screen.refresh()
		


		
		#if profile.matchtype=='dir': paths=[d+'/' for d,_,files in os.walk(profile.parentfolder)]
		#else: paths=['{}/{}/{}'.format(r,d,f) for r,D,F in os.walk(profile.parentfolder) for d in D for f in F]
		#['{}/{}{}'.format(r,'' if len(_d_)==0 else _d_[0]+'/',f) for r,_d_,F in os.walk(profile.parentfolder) for f in F+['']]
		ls=0
		choices=[]		# multiple case
		mode='browse'
		inputtext=''

		allowtextinput=True if 'dynamiccondition' in profile.keys() else False

		while True:
			#matches=filter(profile.options,lambda option: profile.dynamiccondition(option,inputtext))\
			if allowtextinput:
				matches=[option for option in profile.options\
						if profile.dynamiccondition(profile.displayoption(option),inputtext) not in [None,False]]
			else: matches=profile.options

			#explainpad.draw()
			ls=max(0,min(len(matches)-1,ls))
			displayoptions(matches,ls,choices,listpad,matchinfotextdisp,profile,H)
			explanationtextdisp.msg=explanation.format(mode,inputtext)
			explanationtextdisp.draw()

			c=cdisplay.extractkey_cs(screen.getch())

			if c=='ENTER': break

			match mode:
				case 'browse':
					match c:
						case 'SPACE':
							if profile.onlyone: choices.append(ls)
						case 'BACKSPACE':
							if not profile.onlyone and ls in choices: choices.remove(ls)
						case 259: ls-=1
						case 258: ls+=1
						case 260: ls-=5
						case 261: ls+=5
						case 's':
							try: ls=max([c for c in choices if c<ls])
							except: pass
						case 'c':
							try: ls=min([c for c in choices if c>ls])
							except: pass
						case 'i':
							if allowtextinput: mode='input'
						case 'q':
							quit()
						case 'b':
							return None

				case 'input':
					if c=='BACKSPACE': inputtext=inputtext[:-1]
					elif c=='SPACE': inputtext+=' '
					elif c==27: mode='browse'
					else:
						try: inputtext+=c
						except: mode='browse'


		screen.nodelay(True)
		return matches[ls] if profile.onlyone else [matches[ls] for ls in choices]

	@staticmethod
	def getdefaultprofile():
		profile=tracking.Profile(name='browsing')
		profile.msg='select folder'
		profile.onlyone=True
		profile.readinfo=lambda path: sysutil.readtextfile(path+'info.txt')
		profile.options=getpaths(defaultpathprofile())
		profile.msg0='\n\n'\
				+'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
				+'\n\nYou may be able to scroll\nwith the touchpad.'\
				+'\n\nPress ENTER to finish selection.'
		profile.msg1='\n\n'\
				+'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
				+'\n\nYou may be able to scroll\nwith the touchpad.'\
				+'\n\nPress b to continue without selection.'\
				+'\n\nPress ENTER to finish selection.'
		profile.msg2='\n\n'\
				+'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
				+'\n\nYou may be able to scroll\nwith the touchpad.'\
				+'\n\nPress SPACE or a to add (i.e. mark) elements.'\
				+'\nPress s or c to move between marked elements.'\
				+'\n\nPress b to continue without selection.'\
				+'\n\nPress ENTER to finish selection'
		profile.msg=profile.msg1
		profile.displayoption=lambda option:option
		return profile

def displayoptions(options,selection,selections,listpad,matchinfotextdisp,profile,H):
	#matchinfopad.erase()
	listpad.erase()
	for i,match in enumerate(options):
		listpad.addstr(i,2,'{}: {}'.format(str(i+1),profile.displayoption(match)))
	try:
		#matchinfopad.addstr(0,0,getmetadata(options[selection]))
		#matchinfopad.addstr(2,0,getinfo(profile.readinfo,options[selection]))
		matchinfotextdisp.msg=getinfo(profile.readinfo,options[selection])
	except:
		#matchinfopad.addstr(0,0,'no folder selected or no info.txt')
		matchinfotextdisp.msg='no folder selected or no info.txt'
	
	listpad.addstr(selection,0,' *' if profile.onlyone else '>')
	for s in selections: listpad.addstr(s,1,'*')
	listpad.refresh(max(0,selection-H//2))
	matchinfotextdisp.draw()



def combineconditions(profile):
	conditions=[]
	for i in range(1,10):
		cname='condition'+str(i)
		if cname in profile.keys(): conditions.append(profile[cname])
		else: break
	def CONDITION(d):
		try: return all([True if c==None else c(d) for c in conditions])
		except: False
	return CONDITION 	



# for path browsing


def allpaths(root):
	scan=os.scandir(root)
	out=[]
	for d in scan:
		if d.is_file(): out.append(d.name)
		if d.is_dir():
			out.append(d.name+'/')
			out+=[d.name+'/'+branch for branch in allpaths(root+'/'+d.name)]
	return out

def getmetadata(folder):
	try:
		with open(folder+'metadata.txt','r') as f: return ' - '+f.readline()
	except Exception as e: return ''

def getinfo(readinfo,path):
	try: return readinfo(path)
	except: return 'no info'



def getpaths(profile):
	paths=allpaths(profile.parentfolder)
	pattern=re.compile(profile.regex)
	paths=list(filter(pattern.fullmatch,paths))
	paths=list(filter(combineconditions(profile),paths))

	try: paths.sort(reverse=True,key=lambda path:os.path.getmtime(path))
	except: pass

	return paths





#browsingprofile=tracking.Profile()
#
#def pickfolders(**kw):
#	return cdisplay.subtask_in_display(_pickfolders_,browsingprofile,**kw)
#
#def pickfolders_leave_cs(**kw):
#	cdisplay.run_in_display(_pickfolders_,browsingprofile,nodelay=False,**kw)
#	return browsingprofile.loadedpathandpaths
#
#
#
##
#if __name__=='__main__':
#	import cdisplay
#	cdisplay.session_in_display(_pickfolders_,getdefaultprofile())
