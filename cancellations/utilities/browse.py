import os
import re
import curses as cs
import pdb

from cancellations.utilities import textutil, setup

from . import tracking
#from ..display.display import StaticText, dash
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

	profile.options=list(filter(profile.condition,profile.options))
	if len(profile.options)==0: return None


	#setup.screen.nodelay(False)

	L,C,R=display.hsplit(rlimits=[.33,.66])
	C0,C1,Cr=C.hsplit(limits=[2,4],sep=1)

	pointer=tracking.Pointer(val=0)
	selections=[]
	

	explanation=profile.msg
	Ltext=L.add(0,0,_display_._TextDisplay_(explanation))
	C0.add(0,0,_display_._Display_()).encode=lambda: [(0,pointer.val,'>')]
	C1.add(0,0,_display_._Display_()).encode=lambda: [(0,i,'*') for i in selections]
	optionsdisplay=Cr.add(0,0,_display_._TextDisplay_(''))

	getcorner=lambda _: (0,max(pointer.val-display.height//2,0))
	C0.getcorner=getcorner
	C1.getcorner=getcorner
	Cr.getcorner=getcorner



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
		c=setup.getch(lambda: profile.msg)

		if c=='ENTER': break
		if c!=-1:
			match mode:
				case 'browse':
					match c:
						case 'SPACE':
							if profile.onlyone: selections.append(ls)
						case 'BACKSPACE':
							if not profile.onlyone and ls in selections: selections.remove(ls)
						case 'UP': ls-=1
						case 'DOWN': ls+=1
						case 'LEFT': ls-=5
						case 'RIGHT': ls+=5
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



		#ls=max(0,min(len(matches)-1,ls))
		ls=ls%len(matches)
		
		Ltext.msg=explanation.format(mode,inputtext)
		optionsdisplay.msg='\n'.join([profile.displayoption(o) for o in matches])
		Rtext.msg=getinfo(profile.readinfo,profile.options[ls])
		pointer.val=ls
		display.draw()


	#setup.screen.nodelay(True)
	return matches[ls] if profile.onlyone else [matches[ls] for ls in selections]


msg='\n\n'\
	+'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
	+'\nYou may be able to scroll with the touchpad.'\
	+'\n\nPress [i] to input filter phrase (escape with arrow keys).\nmode: {}\nphrase: {}'\
	+'\n\nPress ENTER to finish selection.'

class Browse(_display_.Process):
	processname='browse'
	execprocess=browse

	@staticmethod
	def getdefaultprofile(**kw):
		profile=tracking.Profile(profilename='browsing')
		profile.msg='select'
		profile.onlyone=True
		profile.readinfo=lambda selection: str(selection) #sysutil.readtextfile(path+'info.txt')
		profile.msg=msg
		profile.options=getpaths(defaultpathprofile(**kw))
		profile.displayoption=lambda option:option
		profile.dynamiccondition=lambda fulldotpath,phrase: re.search('.*'.join(phrase),fulldotpath.replace('.',''))
		profile.condition=lambda option:True
		return profile

	@staticmethod
	def getdefaultfilebrowsingprofile(parentfolder='outputs/',**kw):
		profile=tracking.Profile(profilename='file browsing',parentfolder=parentfolder)
		profile.msg='select file'
		profile.onlyone=True
		profile.readinfo=lambda path: sysutil.readtextfile(profile.parentfolder+path+'info.txt')
		profile.options=getpaths(defaultpathprofile(**kw))
		profile.msg=msg
		profile.displayoption=lambda option:option+getmetadata(profile.parentfolder+option)
		profile.dynamiccondition=lambda fulldotpath,phrase: re.search('.*'.join(phrase),fulldotpath.replace('.',''))
		profile.condition=lambda option:True
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
	#return readinfo(path)
	try: return readinfo(path)
	except: return 'no info'





####################################################################################################


def defaultpathprofile(**kw):
	return tracking.Profile(name='filterpaths',\
	parentfolder='outputs/',\
	regex='.*',\
	condition=lambda path:os.path.exists('outputs/'+path+'/data/setup'),\
	).butwith(**kw)

def getpaths(pathprofile):
	paths=allpaths(pathprofile.parentfolder)
	pattern=re.compile(pathprofile.regex)
	paths=list(filter(pattern.fullmatch,paths))
	paths=list(filter(pathprofile.condition,paths))

	paths.sort(reverse=True,key=lambda path:os.path.getmtime(pathprofile.parentfolder+path))
	#try: paths.sort(reverse=True,key=lambda path:os.path.getmtime(pathprofile.parentfolder+path))
	#except: pass

	return paths



