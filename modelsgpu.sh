#!/bin/bash
source activate models
export PYTHONPATH="$PYTHONPATH:/nfs/site/home/jiangyu1/models"
#PARAMS=big
#PARAMS=base
PARAMS=blayers
#SEED=1
TARGET=25
TRAIN_EPOCHS=10
EPOCHS_BETWEEN_EVAL=1
#BATCH_SIZE=2048
#BATCH_SIZE=4096
#CPU_CORES=4

#python_script=/nfs/site/home/jiangyu1/reference/translation/tensorflow/transformer/transformer_main.py
python_script=/nfs/site/home/jiangyu1/models/official/transformer/transformer_main.py
TEST_SRC=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.en
TEST_REF=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.de 

#DATA_DIR=/home/jiangyu/
DATA_DIR=/ubtdata1/jiangyu/models/transformer/data
#MODEL_DIR=/ubtdata1/jiagyu/models/transformer/model_$PARAMS
MODEL_DIR=/ubtdata1/jiangyu/models/transformer/model_bLayersNew
VOCAB_FILE=$DATA_DIR/vocab.ende.32768
#mkdir -p $MODEL_DIR

python $python_script \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --vocab_file=$VOCAB_FILE \
  --param_set=$PARAMS \
  --bleu_source=$TEST_SRC \
  --bleu_ref=$TEST_REF \
  --stop_threshold $TARGET \
  --train_epochs $TRAIN_EPOCHS \
  --epochs_between_evals $EPOCHS_BETWEEN_EVAL \

#  --random_seed=$SEED \
#  --batch_size=$BATCH_SIZE \
#  --num_cpu_cores $CPU_CORES \
#  --intra_op_parallelism_threads=$INTRA \
#  --inter_op_parallelism_threads=$INTER \
#  --kmp_blocktime=$BLOCKTIME \
