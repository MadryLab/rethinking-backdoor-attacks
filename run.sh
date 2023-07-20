#!/bin/bash

DM_PATH="./datamodels.pt"
SAVE_PATH="./results"
NUM_TRIALS=100

# To create the memmaps where to save the results
python -m init_store \
    --cfg.dm_path $DM_PATH \
    --cfg.save_path $SAVE_PATH \
    --cfg.num_trials $NUM_TRIALS

# Compute the scores, and store the trilas in the memmaps
python -m compute_scores \
    --cfg.dm_path $DM_PATH \
    --cfg.save_path $SAVE_PATH \
    --cfg.num_trials $NUM_TRIALS

# Aggregate the final score from the random trials
python -m aggregate_scores \
    --cfg.save_path $SAVE_PATH \
