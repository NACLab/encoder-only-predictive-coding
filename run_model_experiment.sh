#!/bin/bash
## Driver script for running a full set of experimental trials for a given model under a given dataset

### external arguments
GPU_ID=$1 ## point to a GPU (integer identifier)
DATASET=$2 ## dataset to use for simulation
MODEL=$3 ## model-type to simulate
CONFIG=$4 ## model config file (JSON)
EXP_DIR=$5 # "exp_out/"
DATA_DIR=$6 #"../representational-pc/data" #"../../../data/"
TRAIN_MODEL=$7 ## 1 = train model, 0 = do not train model
COLLECT_LATENTS=$8 ## 1 = collect latents, 0 = do not collect latents
DO_KNN_PROBE=${9} ## do KNN probe?
TRAIN_GLIMPSE_POLICY=${10} ## "random", "deterministic", "stochastic"
TRAIN_N_GLIMPSES=${11}
TRAIN_GLIMPSE_BOUNDS=${12}
EVAL_GLIMPSE_POLICY=${13} ## "random", "deterministic", "stochastic"
EVAL_N_GLIMPSES=${14}
EVAL_GLIMPSE_BOUNDS=${15}
USE_SPLITS=${16} ## should data splits (by seed) be used?
SEEDS=( "${@:17}" )

CLEAR_LATENTS=0 ## 1 = clear latent files after usage, 0 = preserve latent files
DATASET_POST_TAG=""
METRIC_TAG=""


### internal arguments
### construct data file-name variables
DATAX="$DATA_DIR/$DATASET/$DATASET_POST_TAG""trainX.npy"
DATAY="$DATA_DIR/$DATASET/$DATASET_POST_TAG""trainY.npy"
TEST_DATAX="$DATA_DIR/$DATASET/testX.npy"
TEST_DATAY="$DATA_DIR/$DATASET/testY.npy"

## go through all seeds/trials
for SEED in "${SEEDS[@]}"; do
  ### using splits/cross-folds requires a different fname tag/naming
  if [ "$USE_SPLITS" -eq 1 ]; then
    DATAX="$DATA_DIR/$DATASET/$DATASET_POST_TAG""split$SEED""_trainX.npy"
    DATAY="$DATA_DIR/$DATASET/$DATASET_POST_TAG""split$SEED""_trainY.npy"
    TEST_DATAX="$DATA_DIR/$DATASET/""split$SEED""_testX.npy"
    TEST_DATAY="$DATA_DIR/$DATASET/""split$SEED""_testY.npy"
  fi

  ### create local latent code pointers
  #echo "################ $EXP_DIR"
  LAT_DIR="$EXP_DIR/$MODEL/$DATASET/$SEED/"
  LAT_FNAME="$DATASET_POST_TAG""latentsX_$EVAL_GLIMPSE_POLICY""_$EVAL_GLIMPSE_BOUNDS"
  TEST_LAT_FNAME="test_latentsX_$EVAL_GLIMPSE_POLICY""_$EVAL_GLIMPSE_BOUNDS"
  LATX="$LAT_DIR/$LAT_FNAME.npy"
  TEST_LATX="$LAT_DIR/$TEST_LAT_FNAME.npy"

  echo " >>>>>>> Executing training process <<<<<<<<"
  echo " >> Trial $SEED model: $CONFIG (on GPU.id: $GPU_ID)"
  echo " >>     under dataset: $DATAX"
  echo " >>     latent dir: $LAT_DIR"

  ### train model
  if [ "$TRAIN_MODEL" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python sim_dynamic_model.py --data_fname=$DATAX --exp_dir=$EXP_DIR \
                                                             --config_fname=$CONFIG --seed=$SEED \
                                                             --n_glimpses=$TRAIN_N_GLIMPSES \
							     --glimpse_bounds=$TRAIN_GLIMPSE_BOUNDS
  fi
  ### collect latents from model
  if [ $COLLECT_LATENTS -eq 1 ]; then
    COMPRESS_LATS=True
    #echo "#################################################"

    #echo ">>> Collecting train latents <<<"
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_dynamic_latents.py --data_fname=$DATAX --exp_dir=$EXP_DIR \
                                                                   --config_fname=$CONFIG \
                                                                   --seed=$SEED --trial_id="$SEED" \
                                                                   --compress_latents=$COMPRESS_LATS \
                                                                   --latents_fname=$LAT_FNAME \
                                                                   --n_glimpses=$EVAL_N_GLIMPSES \
    								   --glimpse_bounds=$EVAL_GLIMPSE_BOUNDS \
                                                                   --glimpse_policy=$EVAL_GLIMPSE_POLICY 
    #echo ">>> Collecting test latents <<<"
    CUDA_VISIBLE_DEVICES=$GPU_ID python extract_dynamic_latents.py --data_fname=$TEST_DATAX --exp_dir=$EXP_DIR \
                                                                   --config_fname=$CONFIG --seed=$SEED --trial_id="$SEED" \
                                                                   --compress_latents=$COMPRESS_LATS \
                                                                   --latents_fname=$TEST_LAT_FNAME \
                                                                   --n_glimpses=$EVAL_N_GLIMPSES \
								   --glimpse_bounds=$EVAL_GLIMPSE_BOUNDS \
                                                                   --glimpse_policy=$EVAL_GLIMPSE_POLICY
  fi
  echo "++++++++++++++++++++++++++++++++++++++++"
  ### run downstream probes on collected model latents
  if [ $DO_KNN_PROBE -eq 1 ]; then ## K-NN classifier probe
    echo ">> Adding metric (acc) post-tag: $METRIC_TAG"
    CUDA_VISIBLE_DEVICES=$GPU_ID python knn_probe_latents.py --seed=$SEED --data_fname=$LATX --lab_fname=$DATAY \
                                                             --test_data_fname=$TEST_LATX --test_lab_fname=$TEST_DATAY \
                                                             --config_fname=$CONFIG --metric_tag=$METRIC_TAG
  fi
  ### do we clear latent code files after downstream probing?
  if [ "$CLEAR_LATENTS" -eq 1 ]; then
    rm $LATX
    #rm "$LAT_DIR/$LAT_FNAME""_distances.npy"
    rm $TEST_LATX
    #rm "$LAT_DIR/$TEST_LAT_FNAME""_distances.npy"
  fi
done
