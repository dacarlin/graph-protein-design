#!/bin/bash

# do 3: coarse, full, and justas (first two from ingraham, second one proteinmpnn)
# establish the importance of the features here via ablation 

python train.py \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --batch_tokens=1000 \
    --features full 
