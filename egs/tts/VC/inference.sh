# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

 
export PYTHONPATH="./"
 


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

# 从这里开始
clash

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_clean.json
fi



echo "Exprimental Configuration File: $exp_config"

# hubertnew="/mnt/petrelfs/hehaorui/data/ckpt/vc/newmhubert/model.safetensors"

hubertold="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/vc_mls_clean/model.safetensors"
whisperold="/mnt/data3/hehaorui/pretrained_models/VC/old_whisper/pytorch_model.bin"
hubert="/mntnfs/lee_data1/caijunwang/ckpt/vc_new_exp/new_mhubert/checkpoint/epoch-0005_step-0009935_loss-1.561842/pytorch_model.bin"
hubert_se="/mnt/petrelfs/hehaorui/data/ckpt/vc/mhubert-noise-se/checkpoint/epoch-0000_step-0080000_loss-1.515860/pytorch_model.bin"
whisper="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper/checkpoint/epoch-0000_step-0400001_loss-1.194134/model.safetensors"
whisper_se="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper_aug/checkpoint/epoch-0000_step-0468003_loss-2.859798/model.safetensors"
whisper_se_spk="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper_aug_spk/checkpoint/epoch-0000_step-0583003_loss-3.672843/model.safetensors"
hubert_se="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/epoch-0001_step-0796000_loss-0.567479/model.safetensors"
hubert_se_both="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert_aug_spk_both/checkpoint/epoch-0001_step-0844000_loss-1.542532/model.safetensors"

#模型的
hubert_clean="/mntnfs/lee_data1/vcdata/model.safetensors"
hubert_ref_noise="xx"
hubert_both_noise="xx"

checkpoint_path=$hubert

# gpu的编号：一般用6/7,换卡
cuda_id=0

#prompt就是reference， target就是ground truth
zero_shot_json_file_path="/mntnfs/lee_data1/vcdata/VCTK/zero_shot_json.json" #测试用例的json文件
output_dir="/mntnfs/lee_data1/vcdata/ckpt/out_test" #
vocoder_path="/mntnfs/lee_data1/vcdata/g_00490000"
wavlm_path="/mntnfs/lee_data1/vcdata/wavlm-base-plus-sv"
#加一个ASR模型的path
#用来算WER


echo "CUDA ID: $cuda_id"
echo "Zero Shot Json File Path: $zero_shot_json_file_path"
echo "Checkpoint Path: $checkpoint_path"
echo "Output Directory: $output_dir"
echo "Vocoder Path: $vocoder_path"
echo "WavLM Path: $wavlm_path"

# both clean
python "${work_dir}"/models/tts/vc/vc_inference.py \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --zero_shot_json_file_path $zero_shot_json_file_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --vocoder_path $vocoder_path \
    --wavlm_path $wavlm_path 

# # 测试的reference是脏的
# python "${work_dir}"/models/tts/vc/vc_inference.py \
#     --config $exp_config \
#     --checkpoint_path $checkpoint_path \
#     --zero_shot_json_file_path $zero_shot_json_file_path \
#     --output_dir $output_dir \
#     --cuda_id ${cuda_id} \
#     --vocoder_path $vocoder_path \
#     --wavlm_path $wavlm_path \
#     --ref_noisy \


# # 测试的source是脏的
# python "${work_dir}"/models/tts/vc/vc_inference.py \
#     --config $exp_config \
#     --checkpoint_path $checkpoint_path \
#     --zero_shot_json_file_path $zero_shot_json_file_path \
#     --output_dir $output_dir \
#     --cuda_id ${cuda_id} \
#     --vocoder_path $vocoder_path \
#     --wavlm_path $wavlm_path \
#     --source_noisy \

# # 测试的source和reference都是脏的
# python "${work_dir}"/models/tts/vc/vc_inference.py \
#     --config $exp_config \
#     --checkpoint_path $checkpoint_path \
#     --zero_shot_json_file_path $zero_shot_json_file_path \
#     --output_dir $output_dir \
#     --cuda_id ${cuda_id} \
#     --vocoder_path $vocoder_path \
#     --wavlm_path $wavlm_path \
#     --ref_noisy \
#     --source_noisy \
 
 
