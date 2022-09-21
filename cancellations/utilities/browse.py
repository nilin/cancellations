import os
import re
import curses as cs
import pdb

from cancellations.utilities import textutil, setup

from . import tracking
from ..display.display import StaticText, dash
from ..display import display as disp, _display_
from ..display import cdisplay
from ..utilities import config as cfg,sysutil,tracking
import time

from cancellations import utilities


up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'








def browse(process):
	profile,display=process.profile,process.display

	L,C,R=display.hsplit(rlimits=[.33,.66])
	C0,C1,Cr=C.hsplit(limits=[2,4],sep=1)

	pointer=tracking.Pointer(val=0)
	selections=[]
	

	explanation=profile.msg
	Ltext=L.add(0,0,_display_._TextDisplay_(explanation))
	C0.add(0,0,_display_._Display_())._getelementstrings_=lambda: [(0,pointer.val,'>')]
	C1.add(0,0,_display_._Display_())._getelementstrings_=lambda: [(0,i,'*') for i in selections]
	Cr.add(0,0,_display_._TextDisplay_('\n'.join([profile.displayoption(o) for o in profile.options])))
	Rtext=matchinfodisp=R.add(0,0,_display_._TextDisplay_(''))

	display.arm()
	display.draw()

	matchinfodisp.msg=textutil.lipsum
	display.draw()

	mode='browse'
	inputtext=''

	#allowtextinput=True if 'dynamiccondition' in profile.keys() else False

	while True:
		matches=[option for option in profile.options\
				if profile.dynamiccondition(profile.displayoption(option),inputtext) not in [None,False]]

		ls=pointer.val
		c=setup.getch()

		if c=='ENTER': break
		match mode:
			case 'browse':
				match c:
					case 'SPACE':
						if profile.onlyone: selections.append(ls)
					case 'BACKSPACE':
						if not profile.onlyone and ls in selections: selections.remove(ls)
					case 259: ls-=1
					case 258: ls+=1
					case 260: ls-=5
					case 261: ls+=5
					case 's':
						try: ls=max([c for c in selections if c<ls])
						except: pass
					case 'c':
						try: ls=min([c for c in selections if c>ls])
						except: pass
					case 'i':
						mode='input'
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

		ls=max(0,min(len(matches)-1,ls))

		Ltext.msg=explanation.format(mode,inputtext)
		Rtext.msg=getinfo(profile.readinfo,profile.options[ls])
		pointer.val=ls
		display.draw()


	#screen.nodelay(True)
	return matches[ls] if profile.onlyone else [matches[ls] for ls in selections]


class Browse(cdisplay.Process):
	processname='browse'
	execprocess=browse

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
		profile.dynamiccondition=lambda fulldotpath,phrase: re.search(phrase,fulldotpath)
		return profile

#def displayoptions(options,selection,selections,listpad,matchinfotextdisp,profile,H):
#	#matchinfopad.erase()
#	listpad.erase()
#	for i,match in enumerate(options):
#		listpad.addstr(i,2,'{}: {}'.format(str(i+1),profile.displayoption(match)))
#	try:
#		#matchinfopad.addstr(0,0,getmetadata(options[selection]))
#		#matchinfopad.addstr(2,0,getinfo(profile.readinfo,options[selection]))
#		matchinfotextdisp.msg=getinfo(profile.readinfo,options[selection])
#	except:
#		#matchinfopad.addstr(0,0,'no folder selected or no info.txt')
#		matchinfotextdisp.msg='no folder selected or no info.txt'
#	
#	listpad.addstr(selection,0,' *' if profile.onlyone else '>')
#	for s in selections: listpad.addstr(s,1,'*')
#	listpad.refresh(max(0,selection-H//2))
#	matchinfotextdisp.draw()



#def combineconditions(profile):
#	conditions=[]
#	for i in range(1,10):
#		cname='condition'+str(i)
#		if cname in profile.keys(): conditions.append(profile[cname])
#		else: break
#	def CONDITION(d):
#		try: return all([True if c==None else c(d) for c in conditions])
#		except: False
#	return CONDITION 	



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





####################################################################################################


def defaultpathprofile():
	return tracking.Profile(name='filterpaths',\
	parentfolder='outputs',\
	regex='.*',\
	condition=lambda path:os.path.exists('outputs/'+path+'/data/setup'),\
	)

def getpaths(pathprofile):
	paths=allpaths(pathprofile.parentfolder)
	pattern=re.compile(pathprofile.regex)
	paths=list(filter(pattern.fullmatch,paths))
	paths=list(filter(pathprofile.condition,paths))

	try: paths.sort(reverse=True,key=lambda path:os.path.getmtime(path))
	except: pass

	return paths



