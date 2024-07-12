#!/bin/bash
#SBATCH -J vc
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -A T00120230002
#SBATCH --gres=gpu:4
#SBATCH --nodelist=pgpu18
#SBATCH --output result_whisper_hubert.out         ## filename of the output

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
    exp_config="${exp_dir}"/exp_config_4gpu_whisper.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="my_hubert_whisper_nof0_mse"

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
    # --checkpoint_path /mntnfs/lee_data1/caijunwang/ckpt/vc_whisper_exp/my_hubert_whisper_nof0_mseloss/checkpoint/epoch-0002_step-0004000_loss-0.837785
    # --checkpoint_path /mntnfs/lee_data1/caijunwang/ckpt/vc_new_exp/new_mhubert/checkpoint/final_epoch-0007_step-0012509_loss-2331.561784