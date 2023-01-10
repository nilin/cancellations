import os
import re
from cancellations.config import config as cfg, tracking

from cancellations.utilities import textutil

from cancellations.config import tracking
from cancellations.display import _display_
from cancellations.config import sysutil


up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'



def browse(process):
    profile,display=process.profile,process.display

    if len(profile.options)==0: return None

    assert(len(profile.options)>0)
    if profile.onlyone and len(profile.options)==1:
        (option,)=profile.options
        return option

    L,C,R=display.hsplit(rlimits=[.33,.66])
    C0,C1,Cr=C.hsplit(limits=[2,4],sep=1)

    pointer=tracking.Pointer(val=0)
    selections=[]
    selectionpositions=[[]]

    explanation=profile.msg
    Ltext=L.add(0,0,_display_._TextDisplay_(explanation))
    C0.add(0,0,_display_._Display_()).encode=lambda: [(0,pointer.val,'>')]
    C1.add(0,0,_display_._Display_()).encode=lambda: [(0,i,'*') for i in selectionpositions[0]]
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

    while True:
        matches=profile.options

        ls=pointer.val
        c=cfg.getch(lambda: profile.msg)

        if c=='ENTER':
            return matches[ls] if profile.onlyone else selections

        if c!=-1:
            match mode:
                case 'browse':
                    match c:
                        case 'SPACE':
                            if not profile.onlyone:
                                selections.append(matches[ls])
                                selectionpositions[0]=[i for i,m in enumerate(matches) if m in selections]
                        case 'd':
                            if (not profile.onlyone) and matches[ls] in selections: selections.remove(matches[ls])
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


        ls=ls%len(matches)
        
        Ltext.msg=explanation
        optionsdisplay.msg='\n'.join([profile.displayoption(o) for o in matches])
        Rtext.msg=getinfo(profile.readinfo,profile.options[ls])
        pointer.val=ls
        display.draw()

def squash(string):
    a=5
    b=50
    if len(string)<=a+b:
        return string
    else:
        return string[:a]+'...'+string[3-b:]

msg='\n\n'\
    +'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
    +'\nYou may be able to scroll with the touchpad.'\
    +'\n\nPress ENTER to select.'

msg2='\n\n'\
    +'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
    +'\nYou may be able to scroll with the touchpad.'\
    +'\n\n\n'\
    +'\n'+50*'-'\
    +'\nPress SPACE to select one of several items.'\
    +'\n'+50*'-'\
    +'\n\n\n\nPress ENTER to finish selection.'



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
        profile.displayoption=lambda option:squash(option)
        return profile



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


def getpaths(parentfolder='outputs/', regex='.*', condition=None):
    if condition is None: condition=lambda path: True
    paths=allpaths(parentfolder)
    pattern=re.compile(regex)
    paths=list(filter(pattern.fullmatch,paths))
    paths=list(filter(condition,paths))
    paths.sort(reverse=True,key=lambda path:os.path.getmtime(os.path.join(parentfolder,path)))
    return paths



