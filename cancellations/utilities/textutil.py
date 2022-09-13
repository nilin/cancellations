import re



BOX='\u2588'
box='\u2592'
dash='\u2015'
infty='\u221E'



def indent(s):
    return '\n'.join(['    '+l for l in s.splitlines()])


####


def sidebyside(*elements,separator=''):
    height=max([len(e.splitlines()) for e in elements])
    Es=[makebox(e,height=height,border=box) for e in elements]
    if separator!='':
        sepbox=makebox(separator,height=height,border=' ')
        Es=('NEXTBOX'+sepbox+'NEXTBOX').join(Es).split('NEXTBOX')
    return '\n'.join([''.join(line) for line in zip(*[E.splitlines() for E in Es])])

def padwidth(S,width,fillstyle=' ',border=''):
    return '\n'.join([border+l+(width-len(l))*fillstyle+border for l in S.splitlines()])

def padheight(S,height):
    h=len(S.splitlines()); d=height-h
    a=d//2; b=d-a;
    return a*'\n'+S+b*'\n'

def makebox(S,width=None,height=None,border=' '):
    if height==None: height=len(S.splitlines())
    if width==None: width=max([len(l) for l in S.splitlines()])
    return addborder(addborder(padwidth(padheight(S,height),width),' '),border)

def addborder(S,border):
    lines=S.splitlines()    
    lines=[border*len(lines[0])]+lines+[border*len(lines[0])]+[border*len(lines[0])]
    return '\n'.join([border+l+border for l in lines])

    
def overwrite(s1,s2):
    '\n'.join([''.join([a if b==' ' else b for (a,b) in zip(l1,l2+' '*len(l1))])\
    for l1,l2 in zip(s1.splitlines(),s2.splitlines())])
    

######

def breakat(s,b):
    return (b+'\n').join(s.split('b'))




def test():
    a='Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit,'
    b='sed do eiusmod \ntempor incididunt \nut labore et dolore \nmagna aliqua'
    print(a)
    print(b)
    print(sidebyside(a,b))
    