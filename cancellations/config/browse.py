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


class Browse(_display_.Process):
    processname='browse'

    def execprocess(self):
        if cfg.display_on:
            out=self.browse_in_display()
        else:
            out=self.browse_nodisplay()
        return out

    def browse_in_display(self):
        profile,display=self.profile,self.dashboard
        P=profile
        options=profile.options
        optionstrings=[P.displayoption(o) for o in options] if P.optionstrings is None else P.optionstrings

        if len(profile.options)==0: return None

        assert(len(profile.options)>0)
        if profile.onlyone and len(profile.options)==1:
            (option,)=profile.options
            return option

        top,maindisplay=display.vsplit(rlimits=[.05])
        top.add(0,0,_display_._TextDisplay_(profile.msg))
        L,C,R=maindisplay.hsplit(rlimits=[.33,.66])
        C0,C1,Cr=C.hsplit(limits=[2,4],sep=1)

        pointer=tracking.dotdict(val=0)
        selections=[]
        selectionpositions=[[]]

        Ltext=L.add(0,0,_display_._TextDisplay_(profile.instr))
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

            ls=pointer.val
            c=tracking.getch(lambda: profile.msg)

            if c=='ENTER':
                if profile.onlyone:
                    return options[ls]
                elif len(selections)>0:
                    return selections

            if c!=-1:
                match mode:
                    case 'browse':
                        match c:
                            case 'SPACE':
                                if not profile.onlyone:
                                    selections.append(options[ls])
                            case 'd':
                                if (not profile.onlyone) and options[ls] in selections: selections.remove(options[ls])
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
                        selectionpositions[0]=[i for i,m in enumerate(options) if m in selections]

                    case 'input':
                        if c=='BACKSPACE': inputtext=inputtext[:-1]
                        elif c=='SPACE': inputtext+=' '
                        elif c==27: mode='browse'
                        else:
                            try: inputtext+=c
                            except: mode='browse'

            ls=ls%len(options)
            optionsdisplay.msg='\n'.join(optionstrings)
            Rtext.msg=optionstrings[ls]
            pointer.val=ls
            display.draw()

    def browse_nodisplay(process):
        P=profile=process.profile
        selections=[]
        def getliststr():
            return '\n'.join(['{}{} | {}'.format('*' if i in selections else ' ',i+1,P.displayoption(o)) for i,o in enumerate(P.options)])

        if P.onlyone:
            print(P.msg)
            print(getliststr())
            return P.options[int(input())-1]
        else:
            while True:
                try: 
                    print(P.msg)
                    print(getliststr())
                    i=input()
                    if i=='': return [P.options[s] for s in selections]
                    else: selections.append(int(i)-1)
                except Exception as e:
                    print(e)

    @staticmethod
    def getprofile(**kw):
        P=tracking.Profile()
        P.onlyone=True
        P.optionstrings=None
        P.displayoption=lambda o:str(o)
        P.update(**kw)

        if P.onlyone:
            P.instr=''\
                +'Move with arrow keys.'\
                +'\nYou may be able to scroll with the touchpad.'\
                +'\n\nPress ENTER to select.'
        else:
            P.instr=''\
                +'Move with arrow keys.'\
                +'\nYou may be able to scroll with the touchpad.'\
                +'\n\n\n'\
                +'\n'+50*'-'\
                +'\nPress SPACE to select one of several items.'\
                +'\n'+50*'-'\
                +'\n\n\n\nPress ENTER to finish selection.'

        if 'msg' not in P: P.msg='select'
        P.update(**kw)
        return P


class Process(_display_.Process):

    def browse(self,*a,**kw):
        return self.subprocess(Browse(*a,**kw))

# for path browsing
#
#
#def allpaths(root):
#    scan=os.scandir(root)
#    out=[]
#    for d in scan:
#        if d.is_file(): out.append(d.name)
#        if d.is_dir():
#            out.append(d.name+'/')
#            out+=[d.name+'/'+branch for branch in allpaths(root+'/'+d.name)]
#    return out
#
#def getmetadata(folder):
#    try:
#        with open(folder+'metadata.txt','r') as f: return ' - '+f.readline()
#    except Exception as e: return ''
#
#def getinfo(readinfo,path):
#    try: return readinfo(path)
#    except: return 'no info'
#
#
#def getpaths(parentfolder='outputs/', regex='.*', condition=None):
#    if condition is None: condition=lambda path: True
#    paths=allpaths(parentfolder)
#    pattern=re.compile(regex)
#    paths=list(filter(pattern.fullmatch,paths))
#    paths=list(filter(condition,paths))
#    paths.sort(reverse=True,key=lambda path:os.path.getmtime(os.path.join(parentfolder,path)))
#    return paths
#
#
#
#