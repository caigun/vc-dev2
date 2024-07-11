import argparse
import torch
import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_model
import librosa
from utils.util import load_config
import os
import json
from models.tts.vc.vc_trainer import VCTrainer
from utils.util import load_config
from models.tts.vc.whisper2speech.ns2_uniamphion import UniAmphionVC
from models.tts.vc.whisper_feature import WhisperNormal
from models.tts.vc.hubert_kmeans import HubertWithKmeans
from models.tts.vc.vc_utils import mel_spectrogram, extract_world_f0, get_pitch_shifted_speech

utteranceid2text = {
    '001': "Please call Stella.",
    '002': "Ask her to bring these things with her from the store.",
    '003': "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
    '004': "We also need a small plastic snake and a big toy frog for the kids.",
    '005': "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
    '006': "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
}

def build_trainer(args, cfg):
    supported_trainer = {
        "VC": VCTrainer,
    }
    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--output_dir", 
        help="output path",
        required=True,
    )
    parser.add_argument(
        "--zero_shot_json_file_path",
        type=str,
        help="Zero shot json file path",
    )
    parser.add_argument(
        "--cuda_id", 
        type=int, 
        default=7, 
        help="Cuda id for training."
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="path to BigVGAN vocoder checkpoint."
    )
    parser.add_argument(
        "--wavlm_path",
        type=str,
        help="path to wavlm vocoder checkpoint."
    )
    parser.add_argument(
        "--ref_noisy",
        action="store_true",
    )
    parser.add_argument(
        "--source_noisy",
        action="store_true",
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
 
    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("local rank", args.local_rank)

    content_extractor = "mhubert"

    if content_extractor=="whisper":
        args.content_extractor = "whisper"
    else:
        args.content_extractor = "mhubert"

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    ckpt_path = args.checkpoint_path
    zero_shot_json_file_path = args.zero_shot_json_file_path
    if args.content_extractor == "mhubert":
        w2v  = HubertWithKmeans()
    elif args.content_extractor == "whisper":
        print("using whisper")
        w2v = WhisperNormal()
    else:
        raise ValueError("Invalid content extractor: {}".format(args.content_extractor))
    
    if "whisper" in args.content_extractor:
        cfg.model.vc_feature.content_feature_dim = 512
    else:
       cfg.model.vc_feature.content_feature_dim = 768
    w2v = w2v.to(device=args.local_rank)
    w2v.eval()

    model = UniAmphionVC(cfg=cfg.model)
    print("loading model")
    if "pytorch_model.bin" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    elif "model.safetensors" in ckpt_path:
        #breakpoint()
        load_model(model, ckpt_path)
        #torch.save(model.state_dict(),"/mntnfs/lee_data1/vcdata/epoch-0002_step-0689002_loss-0.571602/pytorch_model.bin")
        #breakpoint()
    else:
        raise ValueError("Invalid model!")
    print("model loaded")
    model.cuda(args.local_rank)
    model.eval()
    print("loading zero shot json")
    with open(zero_shot_json_file_path, "r") as f:
        zero_shot_json = json.load(f)
    zero_shot_json = zero_shot_json["test_cases"]
    print("length of test cases", len(zero_shot_json))

    utt_dict = {}
    if args.ref_noisy:
        print("using noisy reference")
        args.output_dir = args.output_dir + "_noisyref"
    else:
        print("using clean reference")
    if args.source_noisy:
        print("using noisy source")
        args.output_dir = args.output_dir + "_noisysource"
    else:
        print("using clean source")
    for info in zero_shot_json:
        utt_id = info["uid"] #根据这个判断txt
        utt_dict[utt_id] = {}
        utt_dict[utt_id]["source_speech"] = info["source_wav_path"].replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
        utt_dict[utt_id]["target_speech"] = info["target_wav_path"].replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
        utt_dict[utt_id]["prompt_speech"] = info["prompt_wav_path"].replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
        if args.ref_noisy:
            utt_dict[utt_id]["prompt_speech"] = info["prompt_wav_path"].replace("/prompt/","/promptnoisy2/").replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
        else:
            utt_dict[utt_id]["prompt_speech"] = info["prompt_wav_path"].replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
        if args.source_noisy:
            utt_dict[utt_id]["source_speech"] = info["source_wav_path"].replace("/source/","/sourcenoisy2/").replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
        else:
            utt_dict[utt_id]["source_speech"] = info["source_wav_path"].replace("/mnt/petrelfs/hehaorui/data/datasets/VCTK","/mnt/data2/hehaorui/datasets/VCTK")
    # if output_dir exists, delete it
    if os.path.exists(args.output_dir):
        os.system(f"rm -r {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    test_cases = []

    os.makedirs(f"{args.output_dir}/recon/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/prompt/mel", exist_ok=True)

    temp_id = 0
    all_keys = utt_dict.keys()
    # random sample 20 samples
    # sample_keys = list(utt_dict.keys())[:10]
    sample_keys = list(utt_dict.keys())
    # 应该有30个
    for utt_id in tqdm(sample_keys):
        utt = utt_dict[utt_id]
        # source is the input
        wav_path = utt["source_speech"]
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
        audio = torch.from_numpy(wav).to(args.local_rank)
        audio = audio[None, :]
        
        # target is the ground truth
        tgt_wav_path = utt["target_speech"]
        tgt_wav,_ = librosa.load(tgt_wav_path, sr=16000)
        tgt_wav = np.pad(tgt_wav, (0, 1600 - len(tgt_wav) % 1600))
        tgt_audio = torch.from_numpy(tgt_wav).to(args.local_rank)
        tgt_audio = tgt_audio[None, :]

        # prompt is the reference
        ref_wav_path = utt["prompt_speech"]
        ref_wav,_ = librosa.load(ref_wav_path, sr=16000)
        ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
        ref_audio = torch.from_numpy(ref_wav).to(args.local_rank)
        ref_audio = ref_audio[None, :]

        with torch.no_grad():
            ref_mel = mel_spectrogram(ref_audio)
            tgt_mel = mel_spectrogram(tgt_audio)
            source_mel = mel_spectrogram(audio)
            ref_mel = ref_mel.transpose(1, 2).to(device=args.local_rank)
            ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(args.local_rank).bool()

            # shifted_speech, need_shift = get_pitch_shifted_speech(audio)
            # if not need_shift:
            #     _, content_feature = w2v(audio)
            # else:
            #     print("pitch shifted....")
            #     _, content_feature = w2v(shifted_speech)   

            _, content_feature = w2v(audio)
            content_feature = content_feature.to(device=args.local_rank)

            pitch_raw = extract_world_f0(audio)
            pitch = (pitch_raw - pitch_raw.mean(dim=1, keepdim=True)) / (
                pitch_raw.std(dim=1, keepdim=True) + 1e-6
            )  
            x0 = model.inference(
                content_feature=content_feature,
                pitch=pitch,
                x_ref=ref_mel,
                x_ref_mask=ref_mask,
                inference_steps=200, 
                sigma=1.2,
            )# 150-300 0.95-1.5

            test_case = dict()
            recon_path = f"{args.output_dir}/recon/mel/recon_{utt_id}.npy"
            ref_path = f"{args.output_dir}/target/mel/target_{utt_id}.npy"
            source_path = f"{args.output_dir}/source/mel/source_{utt_id}.npy"
            prompt_path = f"{args.output_dir}/prompt/mel/prompt_{utt_id}.npy"
            
            test_case["recon_ref_wav_path"] = recon_path.replace("/mel/", "/wav/").replace(".npy", ".wav")
            test_case["reference_wav_path"] = ref_path.replace("/mel/", "/wav/").replace(".npy", ".wav")
            test_case["source_wav_path"] = source_path.replace("/mel/", "/wav/").replace(".npy", ".wav")
            test_case["prompt_wav_path"] = prompt_path.replace("/mel/", "/wav/").replace(".npy", ".wav")

            np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
            np.save(prompt_path, ref_mel.transpose(1, 2).detach().cpu().numpy())
            np.save(ref_path, tgt_mel.detach().cpu().numpy())
            np.save(source_path, source_mel.detach().cpu().numpy())
            test_cases.append(test_case)
    del model, w2v, ref_mel, ref_mask, content_feature, x0, ref_audio, tgt_audio, audio, tgt_mel, source_mel
    data = dict()
    data["dataset"] = "recon"
    data["test_cases"] = test_cases

    # total number of test cases
    print("total number of test cases", len(test_cases))

    with open(f"{args.output_dir}/recon.json", "w") as f:
        json.dump(data, f, indent=4)
    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    print("Generating Reconstructed Wav Files")
    os.system(
        f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/recon/mel'} --output_dir={f'{args.output_dir}/recon/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
    )
    print("Generating Target Wav Files")
    os.system(
        f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/target/mel'} --output_dir={f'{args.output_dir}/target/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
    )
    print("Generating Source Wav Files")
    os.system(
        f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/source/mel'} --output_dir={f'{args.output_dir}/source/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
    )
    print("Generating Prompt Wav Files")
    os.system(
        f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/prompt/mel'} --output_dir={f'{args.output_dir}/prompt/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
    )
    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()

    # 就已经有了全部的wav文件
    print("running vc_test.py")
    # sim-o
    os.system(f"python /mntnfs/lee_data1/caijunwang/vc-dev2/models/tts/vc/vc_test.py --wavlm_path={args.wavlm_path} -r={f'{args.output_dir}/prompt/wav'} -d={f'{args.output_dir}/recon/wav'} --gpu {args.cuda_id}")
    # wer
    # gt text ASR model text
    # f'{args.output_dir}/recon/wav' 
    # 用一个asr的模型转录出他的text
    # 

if __name__ == "__main__":
    main()