#!/bin/bash

## clean if repeated
rm -rf $1
mkdir $1

# COLORS:['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
# SHAPES: 'cross', 'square', 'circle', 'ellipse', 'semicircle', 'triangle', 'pentagon']
# SPATIAL: ['above', 'below', 'right', 'left']

python3 exp.py  --log $1 \
                --seed 66 \
                --feature-extractor densenet161 \
                --img-size 64 \
                --addmc-path "./external/addmc" \
                --grm-path "./external/erg-2018-x86-64-0.9.31.dat" \
                --ace-path "./external/ace" \
                --utool-path "./external/Utool-3.1.1.jar" \
                --prop-input-size 1006 \
                --prop-emb-size 5 \
                --prop-supp-size 3 \
                --prop-threshold 0.6 \
                --rels-input-size 2012 \
                --rels-emb-size 5 \
                --rels-supp-size 3 \
                --rels-threshold 0.6 \
                --train-path "./data/shapeworld/train" \
                --train-num-worlds 30 \
                --train-num-ref-exp 5 \
                --train-shuffle \
                --train-evaluation \
                --batch-freq 1 \
                --batch-epochs 100 \
                --batch-size 8 \
                --batch-shuffle \
                --batch-report-freq 10 \
                --test-path "./data/shapeworld/test" \
                --test-num-worlds 10 \
                --test-num-ref-exp 5 \
                --test-evaluation > $1/log.txt
