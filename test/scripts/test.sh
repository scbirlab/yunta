#!/usr/bin/env bash

set -euox pipefail

FILE1=test/outputs/file1.txt
FILE2=test/outputs/file2.txt
mkdir -p $(dirname $FILE1)
echo "$INPUT1" > $FILE1
echo "${INPUTB[@]}" | tr ' ' $'\n' > $FILE2

# interspecies test
INPUT_IS1=test/inputs/crypto/Q5CPK5_CRYPI.a3m
INPUT_IS2=test/inputs/human/EZRI_HUMAN.a3m
yunta dca-single "$INPUT_IS1" \
    --msa2 "$INPUT_IS2" \
    --interspecies \
    --apc \
    --output test/outputs/dca-single-interspecies.tsv \
    --plot test/outputs/dca-single-interspecies
yunta rf2t-single "$INPUT_IS1" \
    --msa2 "$INPUT_IS2" \
    --interspecies \
    --output test/outputs/rf2t-single-interspecies.tsv \
    --plot test/outputs/rf2t-single-interspecies

INPUT_INTER1=test/inputs/Q38361_D29_integrase.a3m
INPUT_INTER2=test/inputs/P9WGF1_Mtb_Mmr.a3m
INPUT1=test/inputs/DYR_YEAST.a3m
INPUT2=test/inputs/CAPZA_YEAST.a3m
INPUTB=(test/inputs/CAPZA_YEAST.a3m test/inputs/WWM1_YEAST.a3m)

yunta dca-single $INPUT_INTER1 -2 $INPUT_INTER2 \
    --apc \
    -o test/outputs/dca-single-inter.tsv \
    --plot test/outputs/dca-single-inter \
    --interspecies

yunta dca-single $INPUT1 -2 $INPUT2 \
    --apc \
    -o test/outputs/dca-single.tsv \
    --plot test/outputs/dca-single 

yunta dca-many $INPUT1 -2 "${INPUTB[@]}" \
    --apc \
    -o test/outputs/dca-many.tsv \
    --plot test/outputs/dca-many
    
yunta dca-single $FILE1 -2 $FILE2 \
    --list-file \
    --apc \
    -o test/outputs/dca-many-list.tsv \
    --plot test/outputs/dca-many-list

if [ $(diff test/outputs/dca-many-list.tsv test/outputs/dca-many.tsv | wc -l) != 0 ]
then 
    >&2 echo "ERROR: List and basic inputs gave different outputs!"
    exit 1
fi

if [ -z $1 ]
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

    yunta af2-single $INPUT1 -2 "${INPUTB[@]}" \
        -o test/outputs/af2-many 
    yunta af2-single $FILE1 -2 $FILE2 \
        --list-file \
        -o test/outputs/af2-many-list
fi
