#!/usr/bin/env bash
#SBATCH --job-name=train-valle-ar            # Job name
#SBATCH --output result.out         ## filename of the output
#SBATCH --nodes=1                   ## Run all processes on a single node
#SBATCH -w, --nodelist=node03             ## Request the pointed node 	
#SBATCH --ntasks=8                  ## number of processes = 20
#SBATCH --cpus-per-task=1           ## Number of CPU cores allocated to each process
#SBATCH --partition=Project         ## Partition name: Project or Debug (Debug is default)

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
base_dir="/nfsmnt/wangcaijun/vc-dev2"
cd $base_dir
pwd
# exp_dir=$(cd `dirname $0`; pwd)
# work_dir=$(dirname $(dirname $(dirname $exp_dir)))
exp_dir=$base_dir/egs/tts/VC
work_dir=$base_dir

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
echo "exp_dir: $exp_dir"
echo "work_dir: $work_dir"

cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_clean.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="new_mhubert_g4"

if [ -z "$gpu" ]; then
    gpu="0,1,2,3"
fi

######## Train Model ###########
echo "Exprimental Name: $exp_name"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 28500 \
"${work_dir}"/bins/tts/train.py \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    # --resume \
    # --resume_type resume \


    # --resume \
    # --resume_type resume \
    # --checkpoint_path /mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0001_step-0400000_loss-0.037989