#nilin
#2022/7

████████████████████████████████████████████████████████████████████████████████████████████████████

Example usage

% python example.py


% python example.py n=6 widths=25,25,50


% python example.py n=6 widths=25,25,50 initfromfile=data/hist 

(requires previos run with same dimensions)

████████████████████████████████████████████████████████████████████████████████████████████████████

% python example.py
			>> Trains


% Ctrl-C (KeyboardInterrupt)
			>> Paused. 


% 'mb'		(Enter)
% '5000'	(Enter)
			>> Minibatch size changed to 5000


% Enter
			>> Trains


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

% python example.py n=6 samples=25000 widths=25,50,50




――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
PAUSE or END training by KeyboardInterrupt (Ctrl-c) 
――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

Option to show plots or change minibatch size when paused





