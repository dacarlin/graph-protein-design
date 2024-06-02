#!/bin/bash

# do 3: coarse, full, and justas (first two from ingraham, second one proteinmpnn)
# establish the importance of the features here via ablation 

#python3 train_s2s.py --cuda --file_data ../data/chain_set.jsonl --file_splits ../data/chain_set_splits.json --batch_tokens=6000 --features dist --name h128_dist

python3 train.py \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --batch_tokens=10_000 \
    --features full --name h128 \
    --device mps 

# python3 train_s2s.py --cuda --file_data ../data/cath/chain_set.jsonl --file_splits ../data/ollikainen/splits_ollikainen.json --batch_tokens=6000 --features full  --mpnn --name h128_full_mpnn_ollikainen

# python3 train_s2s.py --cuda --file_data ../data/cath/chain_set.jsonl --file_splits ../data/ollikainen/splits_ollikainen.json --batch_tokens=6000 --features full --name h128_full_ollikainen