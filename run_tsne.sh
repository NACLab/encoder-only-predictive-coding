#!/bin/bash
## Example usage:  $ ./run_tsne.sh 0 mnist mpc 55

GPU_ID=$1
DATASET=$2
MODEL=$3
SEED=$4
## NOTE: for Norb, one will want to prepare a sub-setted test-set of 2000 samples per class and visualize this
##       as, otherwise, one might run out of memory on their computing system using the scikit learn tSNE model

EXP_DIR="exp_out/"
COMPRESS_LATS=True
N_GLIMPSES=40 #60 #50 #40 #100 #60 #40 #50
GLIMPSE_BOUNDS=0.3 #0.2 #0.95
GLIMPSE_POLICY="deterministic"

DATA_DIR="../representational-pc/data/$DATASET"
TEST_DATAX="$DATA_DIR/testX.npy"
TEST_DATAY="$DATA_DIR/testY.npy"
LAT_DIR="$EXP_DIR/$MODEL/$DATASET/$SEED"
TEST_LAT_FNAME="test_latentsX.npy"
TEST_LATX="$LAT_DIR/$TEST_LAT_FNAME.npy"


CONFIG="configs/$DATASET""_$MODEL.json"
echo " >> Trial $SEED model: $CONFIG"
echo " >>     under dataset: $DATAX"

echo ">> Collecting test latents!"
CUDA_VISIBLE_DEVICES=$GPU_ID python extract_dynamic_latents.py --data_fname=$TEST_DATAX --exp_dir=$EXP_DIR --config_fname=$CONFIG \
                                                               --seed=$SEED --trial_id="$SEED" --compress_latents=$COMPRESS_LATS \
                                                               --latents_fname=$TEST_LAT_FNAME \
                                                               --n_glimpses=$N_GLIMPSES --glimpse_bounds=$GLIMPSE_BOUNDS \
                                                               --glimpse_policy=$GLIMPSE_POLICY 

CUDA_VISIBLE_DEVICES=$GPU_ID python plot_tsne.py --data_fname=$TEST_LATX --labels_fname=$TEST_DATAY --trial_id="$SEED" --config_fname=$CONFIG

