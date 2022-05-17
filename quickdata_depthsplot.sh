nmax=$1
XH=$2
instances=$3
samples=$4

for depth in 3 4 5
do
	for ac in tanh DReLU_normalized 
	do
		python gen_outputs.py $nmax $depth $ac $XH $instances $samples
	done
done
python plot_depths.py $nmax 3 5 $XH
