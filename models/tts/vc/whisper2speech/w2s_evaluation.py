import argparse
import torch
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_model
import librosa
from utils.util import load_config
import os
from models.tts.vc.whisper2speech.ns2_uniamphion import UniAmphionVC
from models.tts.vc.whisper_feature import WhisperNormal
from models.tts.vc.hubert_kmeans import HubertWithKmeans
from models.tts.vc.vc_utils import mel_spectrogram, extract_world_f0, get_pitch_shifted_speech
from models.tts.vc.whisper2speech.w2s_dataset import get_normal_and_whisper
from models.tts.vc.whisper2speech.hubert.w2u import whisper2vector
from evaluation.metrics.similarity.speaker_similarity import extract_speaker_similarity
import random

def mix_audios(audio, tgt_audio, min_segment_length=32000, max_segment_length=64000):
    """
    Mix two audio signals by interleaving segments of varying lengths.
    
    audio: torch.Tensor, shape (1, N)
    tgt_audio: torch.Tensor, shape (1, N)
    min_segment_length: Minimum length of each segment in samples.
    max_segment_length: Maximum length of each segment in samples.
    """
    
    audio = audio.squeeze(0)
    tgt_audio = tgt_audio.squeeze(0)
    
    total_length = audio.shape[0]
    
    mixed_audio = torch.zeros(total_length, device=audio.device)
    
    current_position = 0

    choice = True
    
    while current_position < total_length:
        segment_length = random.randint(min_segment_length, max_segment_length)
        segment_length = min(segment_length, total_length - current_position)
        if choice:
            segment = audio[current_position:current_position + segment_length]
        else:
            segment = tgt_audio[current_position:current_position + segment_length]
        
        choice = choice == False
        
        mixed_audio[current_position:current_position + segment_length] = segment
        current_position += segment_length
    
    return mixed_audio.unsqueeze(0)

def w2v_process(normal_path, ref_path, args, cfg, model, w2v, id, transcript_path, whisper_path=None):
    if whisper_path==None:
        assert 0
        tgt_wav, wav = get_normal_and_whisper(normal_path, temp_path="/mntcephfs/data/wuzhizheng/LibriTTS_whisper_eval/temp")
    else:
        tgt_wav, _ = librosa.load(normal_path, sr=16000)
        wav, _ = librosa.load(whisper_path, sr=16000)
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
            _, content_feature_normal = w2v(tgt_wav)
        elif cfg.trans_exp.content_extractor == 'whubert':
            content_feature = w2v.forward(audio)
            content_feature_normal = w2v.forward(tgt_wav)

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
            inference_steps=1000, 
            sigma=1.2,
        )

        x0_n2n = model.inference(
            content_feature=content_feature_normal,
            pitch=pitch,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=1000, 
            sigma=1.2,
        )

        recon_path = f"{args.output_dir}/recon/mel/{id}.npy"
        ref_path = f"{args.output_dir}/target/mel/{id}.npy"
        source_path = f"{args.output_dir}/source/mel/{id}.npy"
        n2n_path = f"{args.output_dir}/n2n/mel/{id}.npy"
    
        np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
        np.save(ref_path, tgt_mel.detach().cpu().numpy())
        np.save(source_path, source_mel.transpose(1, 2).detach().cpu().numpy())
        np.save(n2n_path, x0_n2n.transpose(1, 2).detach().cpu().numpy())

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
    parser.add_argument(
        "--length",
        type=str,
        default="long",
        help="length choice of the audio (long/short)"
    )
    parser.add_argument(
        "--normal_dataset",
        type=str,
        default="true",
        help="whether use autoconvert the normal speech to whisper as input"
    )
    parser.add_argument(
        "--mix_utterance", 
        type=str, 
        default='false', 
        help="whether use whisper mix with normal utterance as source"
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
    os.makedirs(f"{args.output_dir}/n2n/mel", exist_ok=True)

    os.makedirs(f"{args.output_dir}/recon/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/n2n/wav", exist_ok=True)

    total_files = 0
    for root, dirs, files in os.walk(args.dataset_path):
        # Identify the book directory level based on folder depth
        parts = root.count(os.sep)
        if parts == args.dataset_path.count(os.sep) + 2:  # This identifies the book level in root/speaker/book structure
            if args.normal_dataset=='true':
                wav_files = sorted([f for f in files if f.endswith('.wav')])
            elif args.normal_dataset=='false':
                wav_files = sorted([f for f in files if f.endswith('_whisper.wav')])
            # wav_files = sorted([f for f in files if f.endswith('.wav')])
            if len(wav_files) > 1:  # Ensure there is more than one file to use the first as a reference
                total_files += len(wav_files) - 1

    test_quantity = min(np.inf, total_files)
    processed_files = tqdm(total=total_files)
    file_id = 1  # Initialize file identifier
    strike = 3

    for root, dirs, files in os.walk(args.dataset_path):
        parts = root.count(os.sep)
        if parts == args.dataset_path.count(os.sep) + 2:  # This identifies the book level
            if args.normal_dataset=='true':
                wav_files = sorted([f for f in files if f.endswith('.wav')])
            elif args.normal_dataset=='false':
                wav_files = sorted([f for f in files if f.endswith('_whisper.wav')])
            if len(wav_files) > 1:
                reference_audio = os.path.join(root, wav_files[0]).replace("_whisper.wav", ".wav")
                nfile = 0
                for audio_file in wav_files[1:]:
                    audio_path = os.path.join(root, audio_file)
                    audio, sr =  librosa.load(audio_path)
                    audio_length = len(audio)/sr
                    processed_files.update(1)
                    if args.length == "long":
                        if audio_length < 8:
                            continue
                    elif args.length == "short":
                        if audio_length < 3 or audio_length >= 8:
                            continue
                    else:
                        raise NotImplementedError
                    if args.normal_dataset=='true':
                        transcript_path = audio_path.replace(".wav", ".original.txt")
                        w2v_process(normal_path=audio_path, ref_path=reference_audio, args=args, cfg=cfg, model=model, w2v=w2v, id=str(file_id), transcript_path=transcript_path)
                    elif args.normal_dataset=='false':
                        transcript_path = audio_path.replace("_whisper.wav", ".original.txt").replace("-whisper", "")
                        normal_path = audio_path.replace("_whisper.wav", ".wav")
                        w2v_process(normal_path=normal_path, whisper_path=audio_path, ref_path=reference_audio, args=args, cfg=cfg, model=model, w2v=w2v, id=str(file_id), transcript_path=transcript_path)
                    else:
                        assert NotImplementedError
                    file_id += 1
                    nfile += 1

                    if file_id == test_quantity+1 or nfile == strike+1:
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
        print("Generating n2n Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/wesper-demo/vocoder.py --input {f'{args.output_dir}/n2n/mel'} --output {f'{args.output_dir}/n2n/wav'} --hifigan={args.vocoder_path}"
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
        print("Generating n2n Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/n2n/mel'} --output_dir={f'{args.output_dir}/n2n/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
        )

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()

    speaker_similarity = extract_speaker_similarity(f'{args.output_dir}/recon/wav', f'{args.output_dir}/target/wav')
    speaker_similarity_n2n = extract_speaker_similarity(f'{args.output_dir}/n2n/wav', f'{args.output_dir}/target/wav')
    speaker_similarity_gt = extract_speaker_similarity(f'{args.output_dir}/target/wav', f'{args.output_dir}/target/wav')
    speaker_similarity_source = extract_speaker_similarity(f'{args.output_dir}/source/wav', f'{args.output_dir}/target/wav')
    print(f"Speaker_similarity: {speaker_similarity}")
    print(f"Speaker_similarity_n2n: {speaker_similarity_n2n}")
    print(f"Speaker_similarity_gt: {speaker_similarity_gt}")
    print(f"Speaker_similarity_source: {speaker_similarity_source}")

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
    os.system(
        f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/n2n/wav --dump_folder /mntnfs/lee_data1/caijunwang/evaluation_results --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name n2n"
    )


if __name__ == "__main__":
    main()