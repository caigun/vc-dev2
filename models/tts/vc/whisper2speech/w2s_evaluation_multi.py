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
from models.tts.vc.whisper2speech.w2s_dataset import get_normal_and_whisper,VCDataset
from models.tts.vc.whisper2speech.hubert.w2u import whisper2vector
from evaluation.metrics.similarity.speaker_similarity import extract_speaker_similarity
import random
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import torch
from torch.nn.functional import pad
from tqdm import tqdm
def create_mask(whisper_lengths, max_length, hop_length=320):
    # Create a mask filled with zeros (shape: B x max_length)
    t = calculate_mel_frames(max_length, hop_length)
    mask = torch.zeros((len(whisper_lengths), t), dtype=torch.float)

    for idx, length in enumerate(whisper_lengths):
        # Set the first `length` elements to 1
        mask[idx, :calculate_mel_frames(length)] = 1

    return mask

def collate_fn(batch):
    return batch

def pad_wav(wav, max_length):
    current_length = len(wav)
    padding_length = max_length - current_length
    if padding_length > 0:
        padded_wav = np.pad(wav, (0, padding_length), mode='constant', constant_values=0)
    else:
        padded_wav = wav[:max_length]
    return padded_wav

def w2v_process(batch, args, cfg, model, w2v, device):
    # 打印 batch 的内容以进行调试
    print("Batch content:", batch)

    # 解包批量数据
    normal_paths = [item['normal_path'] for item in batch]
    whisper_paths = [item['whisper_path'] for item in batch]
    ref_paths = [item['ref_path'] for item in batch]
    ids = [item['id'] for item in batch]

    # 加载和预处理音频
    tgt_wavs = [librosa.load(path, sr=16000)[0] for path in normal_paths]
    whisper_wavs = [librosa.load(path, sr=16000)[0] for path in whisper_paths]
    ref_wavs = [librosa.load(path, sr=16000)[0] for path in ref_paths]
    print(ref_paths)
    print(whisper_paths)
    print(normal_paths)
    # 记录每个音频的原始长度
    tgt_lengths = [len(wav) for wav in tgt_wavs]
    whisper_lengths = [len(wav) for wav in whisper_wavs]
    ref_lengths = [len(wav) for wav in ref_wavs]
    print(ref_lengths)
    print(whisper_lengths)
    print(tgt_lengths)
    # 将音频数据转换为Tensor并进行填充
    tgt_audios = [torch.from_numpy(pad_wav(wav, max(tgt_lengths))).unsqueeze(0).to(device) for wav in tgt_wavs]
    whisper_audios = [torch.from_numpy(pad_wav(wav, max(whisper_lengths))).unsqueeze(0).to(device) for wav in whisper_wavs]
    ref_audios = [torch.from_numpy(pad_wav(wav,  max(ref_lengths))).unsqueeze(0).to(device) for wav in ref_wavs]

    # 合并成一个批次
    tgt_audios = torch.cat(tgt_audios, dim=0)
    whisper_audios = torch.cat(whisper_audios, dim=0)
    ref_audios = torch.cat(ref_audios, dim=0)

    with torch.no_grad():
        # 处理每个音频
        if cfg.trans_exp.use_whisper_mix_with_normal:
            audios = mix_audios(whisper_audios, tgt_audios)
        else:
            audios = whisper_audios

        # 获取mel谱图
        ref_mels = mel_spectrogram(ref_audios, hop_size=320).transpose(1, 2).to(device)
        ref_masks = torch.ones(ref_mels.shape[0], ref_mels.shape[1]).to(device).bool()
        source_mels = mel_spectrogram(audios, hop_size=320).transpose(1, 2).to(device)
        tgt_mels = mel_spectrogram(tgt_audios, hop_size=320).transpose(1, 2).to(device)
        x_masks = create_mask(whisper_lengths, max(whisper_lengths), 320).to(device).bool()
        print(x_masks.shape, source_mels.shape, source_mels.shape, tgt_mels.shape, ref_mels.shape)
        # x_masks = None
        # 提取内容特征
        if cfg.trans_exp.content_extractor == 'mhubert':
            _, content_features = w2v(audios)  # semantic (B, T, 768)
        elif cfg.trans_exp.content_extractor == 'whubert':
            content_features = w2v.forward(audios)

        # 提取音高
        if cfg.trans_exp.use_avg_pitch:
            pitches = extract_world_f0(ref_audios)
            pitches = pitches.mean(dim=1, keepdim=True)
        else:
            pitches = None

        # 模型推理
        x0s = model.inference(
            content_feature=content_features,
            pitch=pitches,
            x_ref=ref_mels,
            x_ref_mask=ref_masks,
            inference_steps=1000,
            sigma=1.2,
            x_mask=x_masks
        )
        x0s_n2n=x0s
        # 保存结果
        for i, id in enumerate(ids):
            
            # 裁剪或填充结果以匹配原始长度
            recon_audio = x0s[i].transpose(0, 1).detach().cpu().numpy()
            ref_audio = ref_mels[i].transpose(0, 1).detach().cpu().numpy()
            source_audio = source_mels[i].transpose(0, 1).detach().cpu().numpy()
            tgt_audio = tgt_mels[i].transpose(0, 1).detach().cpu().numpy()
            n2n_audio = x0s_n2n[i].transpose(0, 1).detach().cpu().numpy()
            print(recon_audio.shape[0],recon_audio.shape[1])
            print(ref_audio.shape[0],ref_audio.shape[1])
            
            # 计算每个音频对应的梅尔谱图帧数
            tgt_frames = calculate_mel_frames(tgt_lengths[i])
            ref_frames = calculate_mel_frames(ref_lengths[i])
            whisper_frames = calculate_mel_frames(whisper_lengths[i])

            # 使用目标帧数调整每个梅尔谱图
            recon_audio = adjust_mel_length(recon_audio, whisper_frames)
            ref_audio = adjust_mel_length(ref_audio, calculate_mel_frames(ref_lengths[i], 320))
            source_audio = adjust_mel_length(source_audio, calculate_mel_frames(whisper_lengths[i], 320))
            tgt_audio = adjust_mel_length(tgt_audio,  calculate_mel_frames(tgt_lengths[i], 320))
            n2n_audio = adjust_mel_length(n2n_audio, tgt_frames)
            # print(recon_audio.shape[0],recon_audio.shape[1])
            # print(ref_audio.shape[0],ref_audio.shape[1])
            recon_path = f"{args.output_dir}/recon/mel/{id}.npy"
            tgt_path = f"{args.output_dir}/target/mel/{id}.npy"
            source_path = f"{args.output_dir}/source/mel/{id}.npy"
            n2n_path = f"{args.output_dir}/n2n/mel/{id}.npy"
            ref_path = f"{args.output_dir}/ref/mel/{id}.npy"
            
            np.save(recon_path, recon_audio)
            np.save(tgt_path, tgt_audio)
            np.save(source_path, source_audio)
            np.save(n2n_path, n2n_audio)
            np.save(ref_path, ref_audio)
def calculate_mel_frames(length, hop_length=320):
    """ 计算给定音频长度对应的梅尔谱图帧数 """
    return (length // hop_length) 

def adjust_mel_length(mel_spectrogram, target_frames):
    """ 根据目标帧数调整梅尔谱图的长度 """
    current_frames = mel_spectrogram.shape[1]
    if current_frames < target_frames:
        # 如果当前帧数小于目标帧数，则进行零填充
        padded_mel = np.pad(mel_spectrogram, ((0, 0), (0, target_frames - current_frames)), mode='constant')
    else:
        # 如果当前帧数大于目标帧数，则进行裁剪
        padded_mel = mel_spectrogram[:, :target_frames]
    return padded_mel

class AudioDataset(Dataset):
    def __init__(self, target_wav_path, ref_wav_path, whisper_wav_path):
        self.target_wav_path = target_wav_path
        self.ref_wav_path = ref_wav_path
        self.whisper_wav_path = whisper_wav_path
        self.wav_files = sorted([f for f in os.listdir(target_wav_path) if f.endswith('.wav')], key=lambda x: int(x[:-4]))

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        audio_file = self.wav_files[idx]
        normal_path = os.path.join(self.target_wav_path, audio_file)
        ref_path = os.path.join(self.ref_wav_path, audio_file)
        whisper_path = os.path.join(self.whisper_wav_path, audio_file)
        id = os.path.splitext(audio_file)[0]
        return {
            'normal_path': normal_path,
            'whisper_path': whisper_path,
            'ref_path': ref_path,
            'id': id
        }

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
    parser.add_argument(
        "--hubert_dimension", 
        type=int, 
        default=768, 
        help="The dimension hubert use"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    # dataset = VCDataset(cfg.trans_exp)
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
            cfg.model.vc_feature.content_feature_dim = args.hubert_dimension
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
    os.makedirs(f"{args.output_dir}/ref/mel", exist_ok=True)

    os.makedirs(f"{args.output_dir}/recon/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/n2n/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/ref/wav", exist_ok=True)

    total_files = 0
    
    target_wav_path = os.path.join(args.dataset_path, 'target', 'wav')
    ref_wav_path = os.path.join(args.dataset_path, 'ref', 'wav')
    whisper_wav_path = os.path.join(args.dataset_path, 'source', 'wav')

    dataset = AudioDataset(target_wav_path, ref_wav_path, whisper_wav_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)

    for batch in tqdm(dataloader, total=len(dataloader)):
        w2v_process(batch, args, cfg, model, w2v, device=args.local_rank)
                
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
        print("Generating ref Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/wesper-demo/vocoder.py --input {f'{args.output_dir}/ref/mel'} --output {f'{args.output_dir}/ref/wav'} --hifigan={args.vocoder_path}"
        )
        os.chdir("/mntnfs/lee_data1/caijunwang/vc-dev2")
    elif cfg.trans_exp.content_extractor == "mhubert":
        print("Generating Reconstructed Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/recon/mel'} --output_dir={f'{args.output_dir}/recon/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
        )
        print("Generating Target Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/target/mel'} --output_dir={f'{args.output_dir}/target/wav'} --checkpoint_file=/mntnfs/lee_data1/vcdata/g_00490000 --gpu {args.cuda_id}"
        )
        print("Generating Source Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/source/mel'} --output_dir={f'{args.output_dir}/source/wav'} --checkpoint_file=/mntnfs/lee_data1/vcdata/g_00490000 --gpu {args.cuda_id}"
        )
        print("Generating n2n Wav Files")
        os.system(
            f"python /mntnfs/lee_data1/caijunwang/vc-dev2/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/n2n/mel'} --output_dir={f'{args.output_dir}/n2n/wav'} --checkpoint_file={args.vocoder_path} --gpu {args.cuda_id}"
        )

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()

    speaker_similarity = extract_speaker_similarity(f'{args.output_dir}/recon/wav', f'{args.dataset_path}/target/wav')
    speaker_similarity_n2n = extract_speaker_similarity(f'{args.output_dir}/n2n/wav', f'{args.dataset_path}/target/wav')
    speaker_similarity_gt = extract_speaker_similarity(f'{args.output_dir}/target/wav', f'{args.dataset_path}/target/wav')
    speaker_similarity_source = extract_speaker_similarity(f'{args.output_dir}/source/wav', f'{args.dataset_path}/target/wav')
    print(f"Speaker_similarity: {speaker_similarity}")
    print(f"Speaker_similarity_n2n: {speaker_similarity_n2n}")
    print(f"Speaker_similarity_gt: {speaker_similarity_gt}")
    print(f"Speaker_similarity_source: {speaker_similarity_source}")
    os.system("cd /mntnfs/lee_data1/qjw/code/vc-dev2")
    subject = "wer"
    # --ltr_path {args.output_dir}/transcript.txt
    # --wer_choose 2
    # os.system(
    #     f"sh /mntnfs/lee_data1/qjw/code/vc-dev2/egs/metrics/run.sh --reference_folder {args.dataset_path}/target/wav --generated_folder {args.output_dir}/recon/wav --dump_folder {args.output_dir} --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.dataset_path}/transcript.txt --language english --name recon"
    # )
    # os.system(
    #     f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/source/wav --dump_folder {args.output_dir} --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name source"
    # )
    # os.system(
    #     f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/target/wav --dump_folder {args.output_dir} --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name gt"
    # )
    # os.system(
    #     f"sh egs/metrics/run.sh --reference_folder {args.output_dir}/target/wav --generated_folder {args.output_dir}/n2n/wav --dump_folder {args.output_dir} --metrics \"{subject}\" --fs 16000 --wer_choose 2 --ltr_path {args.output_dir}/transcript.txt --language english --name n2n"
    # )


if __name__ == "__main__":
    main()