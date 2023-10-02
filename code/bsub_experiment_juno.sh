#!/bin/bash
codeDir="code" # the directory where expt.py lives
method="transfer" # the method you want to use to make predictions, can be "transfer", "target-only", or "raw". 
source="REP" # source dataset
target="GDSC" # target dataset 
splitType="random_split" # the method you want to use to split the data: can be 'random_split' or 'sample_split'
holdoutFrac=".8" # fraction of data holdout (aka in the test set)
dataFile="data/rep-gdsc-ctd2-mean-log.csv" # input data
#writeDir="results/$(date +%F)_run_transfer" # directory to write the output of the script to
foldFile="" # defines 10 folds, if splitType = sample_split
nSteps="5" # number of steps to train model for
splitSeed="0" # if using random_split, splitSeed initializes random state; if using sample_split, split seed indexes into the list of folds, read from foldFile
k_for_transfer_list="5,10,15,20,25,30,35,40,45,50" # list of k values to use for transfer learning

for i in ${k_for_transfer_list//,/ }
do
    # call your procedure/other scripts here below
    echo "k=$i"
    k_for_transfer=$i
    writeDir="results/$(date +%F)_run_transfer/s${source}_t${target}/k_$i"
    mkdir -p "$writeDir"
    bsub -n 1 -W 8:00 -R 'span[hosts=1] rusage[mem=8]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
    "$codeDir"/experiment_juno.sh "$codeDir" "$method" "$source" "$target" "$splitType" "$holdoutFrac" "$dataFile" \
    "$writeDir" "$foldFile" "$nSteps" "$splitSeed"


done



