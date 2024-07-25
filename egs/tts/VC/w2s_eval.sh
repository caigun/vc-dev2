# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

 
export PYTHONPATH="./"
 
base_dir="/mntnfs/lee_data1/caijunwang/vc-dev2"
cd $base_dir
pwd

######## Build Experiment Environment ###########

exp_dir=$base_dir/egs/tts/VC
work_dir=$base_dir

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

echo $work_dir
echo $exp_dir
 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

# 从这里开始
clash

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_whisper.json
fi



echo "Exprimental Configuration File: $exp_config"


mhubert_whisper_medium="/mntnfs/lee_data1/caijunwang/ckpt/w2s_with_normal_medium/w2s_medium/checkpoint/final_epoch-0010_step-0259440_loss-20313.045110/pytorch_model.bin"
checkpoint_path=$mhubert_whisper_medium

cuda_id=0

#prompt就是reference， target就是ground truth
output_dir="/mntnfs/lee_data1/vcdata/ckpt/eval_libritts"
# vocoder_path="/mntnfs/lee_data1/vcdata/g_00490000"
vocoder_path="/mntnfs/lee_data1/caijunwang/resources/g_00205000" #hubert from Wesper
wavlm_path="/mntnfs/lee_data1/vcdata/wavlm-base-plus-sv"
dataset_path="/mntcephfs/data/wuzhizheng/LibriTTS/test-clean"
#加一个ASR模型的path
#用来算WER


echo "CUDA ID: $cuda_id"
echo "Zero Shot Json File Path: $zero_shot_json_file_path"
echo "Checkpoint Path: $checkpoint_path"
echo "Output Directory: $output_dir"
echo "Vocoder Path: $vocoder_path"
echo "WavLM Path: $wavlm_path"

# both clean
python "${work_dir}"/models/tts/vc/whisper2speech/w2s_evaluation.py \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --vocoder_path $vocoder_path \
    --dataset_path $dataset_path \
    --input_type normal\
    --wavlm_path $wavlm_path

