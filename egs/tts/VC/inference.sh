# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="vc_inference"

######## Train Model ###########
echo "Exprimental Name: $exp_name"

python "${work_dir}"/models/tts/vc/vc_inference.py \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    --checkpoint_path "/mnt/data2/hehaorui/ckpt/vc/ns2_large_test/checkpoint/epoch-0000_step-0172000_loss-0.037911/model.safetensors" \
    --output_dir "/mnt/data2/hehaorui/vc_test/Results/VCTK" \
    --cuda_id 7 \