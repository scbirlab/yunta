#!/usr/bin/env bash

set -euox pipefail

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

python -c '
from yunta.structs.msa import MSA

def make_stub(f):
    msa = MSA.from_file(f)
    # Write a cropped version — first 60 columns
    with open(f.split(".")[0] + "_stub.a3m", "w") as f:
        for i, line in enumerate(msa.lines):
            line.sequence = line.sequence[:20]
            print(str(line), file=f)
            if i > 40:
                break

make_stub("test/inputs/DYR_YEAST.a3m")
make_stub("test/inputs/CAPZA_YEAST.a3m")

'
yunta af2-single "test/inputs/DYR_YEAST_stub.a3m" -2 "test/inputs/CAPZA_YEAST_stub.a3m" \
    --plot test/outputs/af2-single \
    -o test/outputs/af2-single.tsv
yunta af2-single $INPUT1 -2 "${INPUTB[@]}" \
    --plot test/outputs/af2-single \
    -o test/outputs/af2-single.tsv
yunta af2-single $FILE1 -2 $FILE2 \
    --list-file \
    --plot test/outputs/af2-many-list \
    -o test/outputs/af2-many-list.tsv
