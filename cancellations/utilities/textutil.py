def indent(s):
    return '\n'.join(['    '+l for l in s.splitlines()])

####


def sidebyside(*elements,separator=''):
    height=max([len(e.splitlines()) for e in elements])
    Es=[box_in(e,height).splitlines() for e in elements]
    return '\n'.join([separator.join(line) for line in zip(*Es)])

def padwidth(S,W=None,fillstyle=' ',border=''):
    if W==None: W=max([len(l) for l in S.splitlines()])
    return '\n'.join([border+l+(W-len(l))*fillstyle+border for l in S.splitlines()])

def padheight(S,H):
    h=len(S.splitlines()); d=H-h
    a=d//2; b=d-a;
    return a*'\n'+S+b*'\n'

def box_in(S,height,vborder='',hborder=''):
    return padwidth(padheight(S,height),border=vborder)



######

def breakat(s,b):
    return (b+'\n').join(s.split('b'))




def test():
    a='Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit,'
    b='sed do eiusmod \ntempor incididunt \nut labore et dolore \nmagna aliqua'
    print(a)
    print(b)
    print(sidebyside(a,b))
    