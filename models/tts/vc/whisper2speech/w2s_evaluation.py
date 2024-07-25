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
from utils.util import load_config
from models.tts.vc.whisper2speech.ns2_uniamphion import UniAmphionVC
from models.tts.vc.whisper_feature import WhisperNormal
from models.tts.vc.hubert_kmeans import HubertWithKmeans
from models.tts.vc.vc_utils import mel_spectrogram, extract_world_f0, get_pitch_shifted_speech
from models.tts.vc.whisper2speech.w2s_dataset import get_normal_and_whisper
from models.tts.vc.whisper2speech.hubert.w2u import whisper2vector

def w2v_process(normal_path, ref_path, args, cfg, model, w2v, id, transcript_path):
    tgt_wav, wav = get_normal_and_whisper(normal_path, temp_path="/mntnfs/lee_data1/caijunwang/ckpt/temp")
    ref_wav, _ = librosa.load(ref_path, sr=16000)

    wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
    audio = torch.from_numpy(wav).to(args.local_rank)
    audio = audio[None, :]

    tgt_wav = np.pad(tgt_wav, (0, 1600 - len(tgt_wav) % 1600))
    tgt_audio = torch.from_numpy(tgt_wav).to(args.local_rank)
    tgt_audio = tgt_audio[None, :]

    ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
    ref_audio = torch.from_numpy(ref_wav).to(args.local_rank)
    ref_audio = ref_audio[None, :]

    with torch.no_grad():
        if cfg.trans_exp.content_extractor == "mhubert":
            ref_mel = mel_spectrogram(ref_audio)
            tgt_mel = mel_spectrogram(tgt_audio)
            source_mel = mel_spectrogram(audio).transpose(1, 2)
        elif cfg.trans_exp.content_extractor == "whubert":
            ref_mel = mel_spectrogram(ref_audio, hop_size=320)
            tgt_mel = mel_spectrogram(tgt_audio, hop_size=320)
            source_mel = mel_spectrogram(audio, hop_size=320).transpose(1, 2)
            
        ref_mel = ref_mel.transpose(1, 2).to(device=args.local_rank)
        ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(args.local_rank).bool()

        if cfg.trans_exp.content_extractor == 'mhubert':
            _, content_feature = w2v(audio) # semantic (B, T, 768)
        elif cfg.trans_exp.content_extractor == 'whubert':
            content_feature = w2v.forward(audio)

        content_feature = content_feature.to(device=args.local_rank)

        if cfg.trans_exp.use_avg_pitch:
            pitch = extract_world_f0(ref_audio)
            pitch = pitch.mean(dim=1, keepdim=True)
        else:
            pitch = None 

        x0 = model.inference(
            content_feature=content_feature,
            pitch=pitch,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=200, 
            sigma=1.2,
        )

        recon_path = f"{args.output_dir}/recon/mel/{id}.npy"
        ref_path = f"{args.output_dir}/target/mel/{id}.npy"
        source_path = f"{args.output_dir}/source/mel/{id}.npy"
    
        np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
        np.save(ref_path, tgt_mel.detach().cpu().numpy())
        np.save(source_path, source_mel.transpose(1, 2).detach().cpu().numpy())

        with open(transcript_path, 'r') as file:
            transcript = file.readline().strip()

        with open(f"{args.output_dir}/transcript.txt", 'a') as file:
            file.write(f"{id}.wav|{transcript}\n")



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
        "--dataset_path",
        type=str,
        help="dataset path",
        required=True
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="normal",
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
    
    if args.content_extractor == "mhubert":
        if cfg.trans_exp.content_extractor == "whubert":
            w2v = whisper2vector(cfg, args.local_rank)
        elif cfg.trans_exp.content_extractor == "mhubert":
            w2v = HubertWithKmeans()
            w2v = w2v.to(device=args.local_rank)
            w2v.eval()
    elif args.content_extractor == "whisper":
        print("using whisper")
        w2v = WhisperNormal()
    else:
        raise ValueError("Invalid content extractor: {}".format(args.content_extractor))
    
    if "whisper" in args.content_extractor:
        cfg.model.vc_feature.content_feature_dim = 512
    else:
        if cfg.trans_exp.content_extractor == "whubert":
            cfg.model.vc_feature.content_feature_dim = 256
        elif cfg.trans_exp.content_extractor == "mhubert":
            cfg.model.vc_feature.content_feature_dim = 768
    # w2v = w2v.to(device=args.local_rank)
    # w2v.eval()

    model = UniAmphionVC(cfg=cfg.model)
    print("loading model")
    if "pytorch_model.bin" in ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    elif "model.safetensors" in ckpt_path:
        load_model(model, ckpt_path)
    else:
        raise ValueError("Invalid model!")
    print("model loaded")
    model.cuda(args.local_rank)
    model.eval()
    if os.path.exists(args.output_dir):
        os.system(f"rm -r {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    os.makedirs(f"{args.output_dir}/recon/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/mel", exist_ok=True)

    os.makedirs(f"{args.output_dir}/recon/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/wav", exist_ok=True)

    total_files = 0
    for root, dirs, files in os.walk(args.dataset_path):
        # Identify the book directory level based on folder depth
        parts = root.count(os.sep)
        if parts == args.dataset_path.count(os.sep) + 2:  # This identifies the book level in root/speaker/book structure
            wav_files = sorted([f for f in files if f.endswith('.wav')])
            if len(wav_files) > 1:  # Ensure there is more than one file to use the first as a reference
                total_files += len(wav_files) - 1

    test_quantity = min(100, total_files)
    processed_files = tqdm(total=test_quantity)
    file_id = 1  # Initialize file identifier

    for root, dirs, files in os.walk(args.dataset_path):
        parts = root.count(os.sep)
        if parts == args.dataset_path.count(os.sep) + 2:  # This identifies the book level
            wav_files = sorted([f for f in files if f.endswith('.wav')])
            if len(wav_files) > 1:
                reference_audio = os.path.join(root, wav_files[0])
                nfile = 0
                for audio_file in wav_files[1:]:
                    audio_path = os.path.join(root, audio_file)
                    transcript_path = audio_path.replace(".wav", ".original.txt")
                    w2v_process(normal_path=audio_path, ref_path=reference_audio, args=args, cfg=cfg, model=model, w2v=w2v, id=str(file_id), transcript_path=transcript_path)
                    processed_files.update(1)  # Update the progress bar for each processed file
                    file_id += 1
                    nfile += 1

                    if file_id == test_quantity+1 or nfile == 6:
                        break
                if file_id == test_quantity+1:
                    break
                

    processed_files.close()
                
    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()

    if cfg.trans_exp.content_extractor == "whubert":
        os.chdir("/mntnfs/lee_data1/caijunwang/wesper-demo")
        print("Generating Reconstructed Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/wesper-demo/vocoder.py --input {f'{args.output_dir}/recon/mel'} --output {f'{args.output_dir}/recon/wav'} --hifigan={args.vocoder_path}"
        )
        print("Generating Target Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/wesper-demo/vocoder.py --input {f'{args.output_dir}/target/mel'} --output {f'{args.output_dir}/target/wav'} --hifigan={args.vocoder_path}"
        )
        print("Generating Source Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/wesper-demo/vocoder.py --input {f'{args.output_dir}/source/mel'} --output {f'{args.output_dir}/source/wav'} --hifigan={args.vocoder_path}"
        )
        os.chdir("/mntnfs/lee_data1/caijunwang/vc-dev2")
    elif cfg.trans_exp.content_extractor == "mhubert":
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

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()

    subject = "wer"
    # --ltr_path {args.output_dir}/transcript.txt
    # --wer_choose 2
    os.system(
        f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/recon/wav --dump_folder /mntnfs/lee_data1/caijunwang/evaluation_results --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name recon"
    )
    os.system(
        f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/source/wav --dump_folder /mntnfs/lee_data1/caijunwang/evaluation_results --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name source"
    )
    os.system(
        f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/target/wav --dump_folder /mntnfs/lee_data1/caijunwang/evaluation_results --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name gt"
    )


if __name__ == "__main__":
    main()