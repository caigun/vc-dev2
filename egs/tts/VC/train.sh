#!/bin/bash
#SBATCH -J vc
#SBATCH -p p-RTX2080
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -A T00120230002
#SBATCH --gres=gpu:1
#SBATCH --output result.out         ## filename of the output

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
base_dir="/mntnfs/lee_data1/caijunwang/vc-dev2"
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

exp_name="new_mhubert"

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