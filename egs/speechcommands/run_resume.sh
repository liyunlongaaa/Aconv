#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
##SBATCH -p sm
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
if [ $HOSTNAME == "yoos-X17-AT-22" ];  then  
  source /home/yoos/anaconda3/bin/activate ast    #本机操作
else
  source ../../venvast/bin/activate
fi  

export TORCH_HOME=../../pretrained_models

model=Conv
dataset=speechcommands
imagenetpretrain=True
audiosetpretrain=False
bal=none
lr=2.5e-4
epoch=31
freqm=48   #mask
timem=48
mixup=0.6
batch_size=32
fstride=4
tstride=4
tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json
exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-demo
model_size="tiny"
resume=True
last_epoch=30

if [ -d $exp_dir ]; then
  echo 'exp exist'
  #exit
fi
mkdir -p $exp_dir

python ./prep_sc.py

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/myrun.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
--model_size $model_size --resume $resume --last_epoch $last_epoch > $exp_dir/log_ConvNeXt.txt
