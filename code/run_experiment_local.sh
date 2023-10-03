#!/bin/bash
codeDir="/Users/jinrui.liu/Desktop/tanseyLab/transfer_learning/transfer-learning/code" # the directory where expt.py lives
method="transfer" # the method you want to use to make predictions, can be "transfer", "target-only", or "raw". 
source="REP" # source dataset
target="GDSC" # target dataset 
splitType="random_split" # the method you want to use to split the data: can be 'random_split' or 'sample_split'
holdoutFrac=".8" # fraction of data holdout (aka in the test set)
dataFile="/Users/jinrui.liu/Desktop/tanseyLab/transfer_learning/transfer-learning/data/drug/rep-gdsc-ctd2-mean-log.csv" # input data
writeDir="/Users/jinrui.liu/Desktop/tanseyLab/transfer_learning/transfer-learning/results/$(date +%F)/test_transfer" # directory to write the output of the script to
foldFile="" # defines 10 folds, if splitType = sample_split
nSteps="5" # number of steps to train model for
splitSeed="0" # if using random_split, splitSeed initializes random state; if using sample_split, split seed indexes into the list of folds, read from foldFile
k_for_transfer="5" # k value to use for transfer learning

mkdir -p "$writeDir"

python3 "$codeDir"/experiment.py method="$method" source="$source" target="$target" splitType="$splitType" holdoutFrac="$holdoutFrac" dataFile="$dataFile" \
writeDir="$writeDir" foldFile="$foldFile" nSteps="$nSteps" splitSeed="$splitSeed" k_for_transfer="$k_for_transfer" 2>&1 | tee "$writeDir"/log.txt



