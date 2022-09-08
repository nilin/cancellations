# source batch.sh $iterations $nprofiles

for i in {0..$2}
do
	python batchprofiles.py $i $1 loadtarget
done
