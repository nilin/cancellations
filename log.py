def log(msg,loud=False):
	with open('log','a') as f:
		f.write('\n'+str(msg))
	if loud:
		print(msg)
