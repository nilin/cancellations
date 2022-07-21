

----------------------------------------------------------------------------------------------------
Example run for learning a Slater of Hermite polynomials 
----------------------------------------------------------------------------------------------------

% python example.py





----------------------------------------------------------------------------------------------------
Force parameter updates with named definitions from command line
----------------------------------------------------------------------------------------------------

% python example.py n=5 trainsamples=20000 batchsize=1000 widths=25,50,100




----------------------------------------------------------------------------------------------------
Params
----------------------------------------------------------------------------------------------------

widths prescribes layer widths. len(widths) is the number of internal layers




----------------------------------------------------------------------------------------------------
PAUSE or END training by KeyboardInterrupt (Ctrl-c). 
----------------------------------------------------------------------------------------------------

Option to show plots or change batch mode.




----------------------------------------------------------------------------------------------------
batchmode
----------------------------------------------------------------------------------------------------

minibatch:	update weights for each minibatch

batch:		Still iterates through minibatches to limit memory usage. Only updates weights after full epoch.
		Equivalent with batch gradient descent. 
