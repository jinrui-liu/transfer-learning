#!/bin/bash
codeDir="code" # the directory where expt.py lives
method="transfer" # the method you want to use to make predictions, can be "transfer", "target-only", or "raw". 
source="REP" # source dataset
target="GDSC" # target dataset 
splitType="random_split" # the method you want to use to split the data: can be 'random_split' or 'sample_split'
holdoutFrac=".8" # fraction of data holdout (aka in the test set)
dataFile="data/rep-gdsc-ctd2-mean-log.csv" # input data
writeDir="results/$(date +%F)/test_transfer" # directory to write the output of the script to
foldFile="fold_info/fold_list.pkl" # defines 10 folds, if splitType = sample_split
nSteps="5" # number of steps to train model for
splitSeed="0" # if using random_split, splitSeed initializes random state; if using sample_split, split seed indexes into the list of folds, read from foldFile
k_for_transfer_list="5,10,15,20,25,30,35,40,45,50" # list of k values to use for transfer learning

mkdir -p "$writeDir"

for i in ${k_for_transfer_list//,/ }
do
    # call your procedure/other scripts here below
    echo "k=$i"
    k_for_transfer=$i
    python3 "$codeDir"/expt.py method="$method" source="$source" target="$target" splitType="$splitType" holdoutFrac="$holdoutFrac" dataFile="$dataFile" \
writeDir="$writeDir" foldFile="$foldFile" nSteps="$nSteps" splitSeed="$splitSeed" k_for_transfer="$k_for_transfer"

done



