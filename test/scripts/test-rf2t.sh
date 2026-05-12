#!/usr/bin/env bash

set -eox pipefail

INPUT_INTER1=test/inputs/Q38361_D29_integrase.a3m
INPUT_INTER2=test/inputs/P9WGF1_Mtb_Mmr.a3m
INPUT1=test/inputs/DYR_YEAST.a3m
INPUT2=test/inputs/CAPZA_YEAST.a3m
INPUTB=(test/inputs/CAPZA_YEAST.a3m test/inputs/WWM1_YEAST.a3m)

FILE1=test/outputs/file1.txt
FILE2=test/outputs/file2.txt
mkdir -p $(dirname $FILE1)
echo "$INPUT1" > $FILE1
echo "${INPUTB[@]}" | tr ' ' $'\n' > $FILE2

# interspecies test
INPUT_IS1=test/inputs/crypto/Q5CPK5_CRYPI.a3m
INPUT_IS2=test/inputs/human/EZRI_HUMAN.a3m

yunta rf2t-single "$INPUT_IS1" \
    --msa2 "$INPUT_IS2" \
    --interspecies \
    --output test/outputs/rf2t-single-interspecies.tsv \
    --plot test/outputs/rf2t-single-interspecies

if [ -z "$1" ]
then
    yunta rf2t-single $INPUT1 -2 $INPUT2 \
        -o test/outputs/rf2t-single.tsv \
        --plot test/outputs/rf2t-single 
    yunta rf2t-single $INPUT1 -2 "${INPUTB[@]}"\
        -o test/outputs/rf2t-many.tsv \
        --plot test/outputs/rf2t-many 
    yunta rf2t-single $FILE1 -2 $FILE2 \
        -o test/outputs/rf2t-many-list.tsv \
        --plot test/outputs/rf2t-many-list \
        --list-file 

    if [ $(diff test/outputs/rf2t-many-list.tsv test/outputs/rf2t-many.tsv | wc -l) != 0 ]
    then 
        >&2 echo "ERROR: List and basic inputs gave different outputs!"
        exit 1
    fi
fi
