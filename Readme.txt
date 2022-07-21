#nilin
#2022/7

████████████████████████████████████████████████████████████████████████████████████████████████████

Example usage

████████████████████████████████████████████████████████████████████████████████████████████████████

% python example.py
	or
% python example.py widths=25,50,100 batchmode=minibatch
	>> Trains minibatch g.d.


% Ctrl-C (KeyboardInterrupt)
	>> Paused. 


% 'b' (Enter)
	>> Batch mode changed to batch g.d.


% Enter
	>> Trains batch g.d.


% Ctrl-C
	>> Paused


% 'p' (Enter)
	>> Show plots. (These are saved anyway)


% 'q' (Enter)
	>> End




████████████████████████████████████████████████████████████████████████████████████████████████████

Explanations

████████████████████████████████████████████████████████████████████████████████████████████████████


――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
Force parameter updates with named definitions from command line
――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

% python example.py

% python example.py n=5 trainsamples=20000 batchsize=1000 widths=25,50,100



――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
Params
――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

widths prescribes layer widths. len(widths) is the number of internal layers


batchmode
....................................................................................................

minibatch:	update weights for each minibatch

batch:		Still iterates through minibatches to limit memory usage. 
		Only updates weights after full epoch.
		Equivalent with batch gradient descent. 


I expect minibatch to work better initially and batch to work better when close to convergence.



――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
PAUSE or END training by KeyboardInterrupt (Ctrl-c) 
――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

Option to show plots or change batch mode when paused





