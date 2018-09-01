#!/bin/bash
source activate tfhvdcpu
TARGET=25
SEED=1
INTRA=28
INTER=2
BLOCKTIME=1
TRAIN_EPOCHS=10
EPOCHS_BETWEEN_EVAL=1

python_script=/nfs/site/home/jiangyu1/reference/translation/tensorflow/transformer/transformer_main_cpu.py
TEST_SRC=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.en
TEST_REF=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.de 

PARAMS=base
DATA_DIR=/home/jiangyu
MODEL_DIR=/ubtdata1/jiangyu/mlperf/transformer/model_$PARAMS-cpu
mkdir -p $DATA_TEST_DIR $DATA_DIR $MODEL_DIR

python transformer_main_cpu.py \
  --random_seed=$SEED \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --params=$PARAMS \
  --bleu_source=$TEST_SRC \
  --bleu_ref=$TEST_REF \
  --intra_op_parallelism_threads=$INTRA \
  --inter_op_parallelism_threads=$INTER \
  --bleu_threshold $TARGET \
  --train_epochs $TRAIN_EPOCHS \
  --epochs_between_eval $EPOCHS_BETWEEN_EVAL 

