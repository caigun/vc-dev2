#!/bin/bash
#SBATCH -J test
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -A T00120230002
#SBATCH --gres=gpu:1
#SBATCH --output /mntnfs/lee_data1/caijunwang/vc-dev2/infer.out

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


mhubert_whisper_medium="/mntnfs/lee_data1/caijunwang/ckpt/w2s_with_normal_medium/w2s_medium_noise/checkpoint/final_epoch-0020_step-0540384_loss-71958.440633/pytorch_model.bin"
checkpoint_path=$mhubert_whisper_medium

cuda_id=0

#prompt就是reference， target就是ground truth
output_dir="/mntcephfs/data/wuzhizheng/LibriTTS_whisper_eval/out/long_1000"
# vocoder_path="/mntnfs/lee_data1/vcdata/g_00490000"
vocoder_path="/mntnfs/lee_data1/caijunwang/resources/g_00205000" #hubert from Wesper
wavlm_path="/mntnfs/lee_data1/vcdata/wavlm-base-plus-sv"
dataset_path="/mntcephfs/data/wuzhizheng/LibriTTS_whisper_eval/test-clean"


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

echo "python convert.py --input ${output_dir}/source/wav --output ${output_dir}/wesper_recon/wav --hubert /mntnfs/lee_data1/caijunwang/resources/model-layer12-450000.pt --fastspeech2 /mntnfs/lee_data1/caijunwang/resources/googletts_neutral_best.tar --hifigan $vocoder_path"
echo "sh egs/metrics/run.sh --reference_folder ${output_dir}/target/wav --generated_folder ${output_dir}/wesper_recon/wav --dump_folder /mntnfs/lee_data1/caijunwang/evaluation_results --metrics "wer" --fs 16000 --wer_choose 2 --ltr_path ${output_dir}/transcript.txt --language english --name wesper"
echo "from evaluation.metrics.similarity.speaker_similarity import extract_speaker_similarity"
echo "extract_speaker_similarity(\"${output_dir}/target/wav\", \"${output_dir}/wesper_recon/wav\")"