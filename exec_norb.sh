#!/bin/bash

GPU_ID=1 ## point to a GPU (integer identifier)
DATASET=norb ## dataset to use for simulation
MODEL=mpc ## model-type to simulate
CONFIG=configs/norb_mpc.json ## model config file (JSON)
EXP_DIR=exp_out/
DATA_DIR=../representational-pc/data/ #../../../data/
TRAIN_MODEL=1 ## 1 = train model, 0 = do not train model
COLLECT_LATENTS=1 ## 1 = collect latents, 0 = do not collect latents
DO_KNN_PROBE=1 ## do KNN probe?
USE_SPLITS=0

SEEDS=(42) #(42 43 44)

## train-time guidelines
TRAIN_GLIMPSE_POLICY="deterministic" ## NOTE: does nothing
TRAIN_N_GLIMPSES=15
TRAIN_GLIMPSE_BOUNDS=0.3
## eval-time guidelines (when collecting latent codes)
EVAL_GLIMPSE_POLICY="deterministic"
EVAL_N_GLIMPSES=40
EVAL_GLIMPSE_BOUNDS=0.3 

mkdir $EXP_DIR 
## drive experiment script
./run_model_experiment.sh $GPU_ID $DATASET $MODEL $CONFIG $EXP_DIR $DATA_DIR \
                          $TRAIN_MODEL $COLLECT_LATENTS $DO_KNN_PROBE \
			  $TRAIN_GLIMPSE_POLICY $TRAIN_N_GLIMPSES $TRAIN_GLIMPSE_BOUNDS \
			  $EVAL_GLIMPSE_POLICY $EVAL_N_GLIMPSES $EVAL_GLIMPSE_BOUNDS \
			  $USE_SPLITS "${SEEDS[@]}"

