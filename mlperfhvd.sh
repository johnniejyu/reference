#!/bin/bash
CPUGPU=$1
if [ "$1" == "cpu" ]
then
  echo 'Train on cpu'
elif [ "$1" == "gpu" ]
then
  echo 'Train on gpu'
else
  echo 'Train on cpu or gpu?'
fi

echo "source activate tfhvd$1"
source activate tfhvd$1
#source activate tfhvddev$1
#cd /nfs/site/home/jiangyu1/horovod
#python setup.py sdist
#pip install --no-cache-dir dist/horovod*.tar.gz
#cd


#--- model setting
PARAMS=big
#BATCH_SIZE=2048
#BATCH_SIZE=1024
TARGET=25
#export BATCH_SIZE=4096 
# priority target > train_steps > train_epochs
TRAIN_STEPS=500000
STEPS_BETWEEN_EVAL=10000
#TRAIN_EPOCHS=10
#EPOCHS_BETWEEN_EVAL=1
CPU_CORES=4

#--- horovod setting
if [ "$CPUGPU" = "cpu" ]
then
  NODES=16
#  NODES=4
  NPERNODE=2
else [ "$CPUGPU" = "gpu" ]
  NODES=1
  NPERNODE=1
fi

HOSTFILE=/nfs/site/home/jiangyu1/workspace/hostfile
#HOSTFILE=/nfs/site/home/jiangyu1/workspace/hostfile$CPUGPU

#--- CPU MKL setting
INTRA=28 #728 #29*26# cores per socket
INTER=2 #52 #2*26
BLOCKTIME=0

#--- TF setting
SEED=1

python_script=/nfs/site/home/jiangyu1/reference/translation/tensorflow/transformer/transformer_main_hvd.py
TEST_SRC=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.en
TEST_REF=/ubtdata1/jiangyu/mlperf/transformer/data_test/newstest2014.de 
DATA_DIR=/home/jiangyu/
MODEL_DIR=/ubtdata1/jiangyu/mlperf/transformer/model_$PARAMS-$CPUGPU-$NODES
HOROVOD_TIMELINE_FILE=$MODEL_DIR/timeline.json
mkdir -p $DATA_DIR $MODEL_DIR

#cd $DATA_TEST_DIR
#wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
#wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
#python data_download.py --data_dir=$DATA_DIR

# use this line to replace python for cpu training 
#  /ubtdata1/jiangyu/miniconda3/envs/tfhvdcpu/bin/python $python_script --random_seed=$SEED \
#  /ubtdata1/jiangyu/miniconda3/envs/tfhvdcpu/bin/python $python_script --random_seed=$SEED \
#  python $python_script --random_seed=$SEED \

HOROVOD_TIMELINE=$HOROVOD_TIMELINE_FILE mpirun --mca btl_tcp_if_include eno1 --bind-to none \
  --hostfile $HOSTFILE \
  -npernode $NPERNODE \
  -n $NODES \
  python $python_script --random_seed=$SEED \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --params=$PARAMS \
  --bleu_source=$TEST_SRC \
  --bleu_ref=$TEST_REF \
  --intra_op_parallelism_threads=$INTRA \
  --inter_op_parallelism_threads=$INTER \
  --kmp_blocktime=$BLOCKTIME \
  --num_cpu_cores $CPU_CORES \
  --bleu_threshold $TARGET \
  --train_steps $TRAIN_STEPS \
  --steps_between_eval $STEPS_BETWEEN_EVAL 

#  --train_epochs $TRAIN_EPOCHS \
#  --epochs_between_eval $EPOCHS_BETWEEN_EVAL \
#  --batch_size=$BATCH_SIZE \
#/ubtdata1/jiangyu/miniconda3/envs/tfcpuhvd/bin/python $python_script --random_seed=$SEED \
#/ubtdata1/jiangyu/miniconda3/envs/tfgpu/bin/python $python_script --random_seed=$SEED \



# Horovod cmds
#mpirun -np $NODES \
#  -H sc12ssgmlt09:2,sc12ssgmlt10:2 \
#  -bind-to none -map-by slot \
#  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#  -mca pml ob1 -mca btl ^openib \
#  /ubtdata1/jiangyu/miniconda3/envs/tfcpuhvd/bin/python $python_script --random_seed=$SEED \
#  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --params=$PARAMS \
#  --bleu_source=$TEST_SRC --bleu_ref=$TEST_REF \
#  --intra_op_parallelism_threads=$INTRA \
#  --inter_op_parallelism_threads=$INTER \
#  --kmp_blocktime=$BLOCKTIME \
#  --batch_size=$BATCH_SIZE \
#  --bleu_threshold $TARGET
#  --train_steps=30000

# example to run on 64-node (2 MPI per node), where each node is Intel Xeon Gold 6148, the distributed training can be launched as
# note that if you want to train models to achieve good accuracy please use –distortions=True. You may also need to change the other hyper-parameters
# https://ai.intel.com/using-intel-xeon-for-multi-node-scaling-of-tensorflow-with-horovod/ 
#export LD_LIBRARY_PATH=<path to OpenMP lib>:$LD_LIBRARY_PATH
#export PATH=<path to OpenMPI bin>:$PATH
#mpirun -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -cpus-per-proc 20 --map-by node  \
#  --report-bindings -hostfile host_names -n 128 \
#  python $python_script --mkl=True \
#  --forward_only=False --num_batches=200 --kmp_blocktime=0 --num_warmup_batches=50 \
#  --intra_op_parallelism_threads=$INTRA --batch_size=$batch_size \
#  --inter_op_parallelism_threads=$INTER --model=$MODEL --variable_update horovod \
#  --horovod_device cpu --data_dir $DATA_DIR --model_dir $MODEL_DIR

