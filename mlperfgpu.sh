#!/bin/bash
#source activate tfhvdgpu
source activate tfhvddevgpu
PARAMS=base
#PARAMS=big
#PARAMS=blayers
SEED=1
TARGET=25
TRAIN_EPOCHS=10
EPOCHS_BETWEEN_EVAL=1
BATCH_SIZE=2048
#BATCH_SIZE=4096
CPU_CORES=4

python_script=/nfs/site/home/jiangyu1/reference/translation/tensorflow/transformer/transformer_main.py
TEST_SRC=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.en
TEST_REF=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.de 

DATA_DIR=/home/jiangyu
#MODEL_DIR=/ubtdata1/jiangyu/mlperf/transformer/skipmodel_$PARAMS-1gpu
#MODEL_DIR=/ubtdata1/jiangyu/models/transformer/model_bLayers
MODEL_DIR=/ubtdata1/jiangyu/mlperf/transformer/model_$PARAMS-1gputest
mkdir -p $MODEL_DIR

python $python_script \
  --random_seed=$SEED \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --params=$PARAMS \
  --bleu_source=$TEST_SRC \
  --bleu_ref=$TEST_REF \
  --num_cpu_cores $CPU_CORES \
  --batch_size=$BATCH_SIZE \
  --bleu_threshold $TARGET \
  --train_epochs $TRAIN_EPOCHS \
  --epochs_between_eval $EPOCHS_BETWEEN_EVAL \

#  --intra_op_parallelism_threads=$INTRA \
#  --inter_op_parallelism_threads=$INTER \
#  --kmp_blocktime=$BLOCKTIME \
