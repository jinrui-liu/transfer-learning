#!/bin/bash
codeDir="$1"
method="$2"
source=$3
target="$4"
splitType="$5"
holdoutFrac="$6"
dataFile="$7"
writeDir="$8"
foldFile="$9"
nSteps="${10}"
splitSeed="${11}"
k_for_transfer="${12}"

source /home/liuj11/miniconda3/etc/profile.d/conda.sh
conda activate tl_env

python3 "$codeDir"/experiment.py method="$method" source="$source" target="$target" splitType="$splitType" holdoutFrac="$holdoutFrac" dataFile="$dataFile" \
writeDir="$writeDir" foldFile="$foldFile" nSteps="$nSteps" splitSeed="$splitSeed" k_for_transfer="$k_for_transfer"