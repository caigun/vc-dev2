import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import (
    BaseCollator,
)
from multiprocessing import Pool, Lock
import random
import torchaudio
import rir_generator as rir
import time
import ctypes
# from models.tts.vc.whisper2speech.s2w import s2w

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
    # if total_length < max_segment_length:
    #     return audio
    
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
        
        # print(audio.shape, tgt_audio.shape, segment.shape, current_position, current_position + segment_length, segment_length, mixed_audio.shape)

        mixed_audio[current_position:current_position + segment_length] = segment
        current_position += segment_length
    
    return mixed_audio

NUM_WORKERS = 64
lock = Lock()  # 创建一个全局锁
SAMPLE_RATE = 16000
lib=ctypes.CDLL("/mntnfs/lee_data1/caijunwang/vc-dev2/models/tts/vc/whisper2speech/toWhisper/libtoWhisper.so")
def process_audio(normal_path,output_path,random_parameter=False):
    input_path = normal_path
    output_file = output_path 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Create random parameters
    if random_parameter:
        l = random.random()*0.99
        r = (random.random()-0.5)*0.3
    else:
        l = 0.6
        r = 0
    # Create argv parameters
    argv = [b"./toWhisper", b"-o", output_file.encode(), b"-l", str(l).encode(), b"-r", str(r).encode(),input_path.encode()]
    argc = len(argv)
    argv_array = (ctypes.POINTER(ctypes.c_char) * (argc + 1))()
    for i, arg in enumerate(argv):
        argv_array[i] = ctypes.create_string_buffer(arg)
    argv_array[argc] = None
    
    # Call genwave function
    result = lib.main(argc, argv_array)
    if result != 0:
        print(f"Failed to process {input_path}: {result}")
    else:
        return input_path
def get_normal_and_whisper(normal_path, temp_path, random_param_in_toWhisper=False):
    filename = os.path.basename(normal_path)
    output_path = os.path.join(temp_path, filename)
    process_audio(normal_path, output_path, random_param_in_toWhisper)
    speech, _ = librosa.load(normal_path, sr=SAMPLE_RATE)
    wspeech, _ = librosa.load(output_path, sr=SAMPLE_RATE)
    os.remove(output_path)
    return speech, wspeech

def get_metadata(file_path):
    metadata = torchaudio.info(file_path)
    return file_path, metadata.num_frames

def get_speaker(file_path):
    speaker_id = file_path.split(os.sep)[-3]
    if 'mls' in file_path:
        speaker = 'mls_' + speaker_id
    else:
        speaker = 'libri_' + speaker_id
    return file_path, speaker

def safe_write_to_file(data, file_path, mode='w'):
    try:
        with lock, open(file_path, mode, encoding='utf-8') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")


class VCDataset(Dataset):
    def __init__(self, args, TRAIN_MODE=True):
        print(f"Initializing VCDataset")
        if TRAIN_MODE:
            directory_list = args.directory_list
        else:
            directory_list = args.test_directory_list
        random.shuffle(directory_list)

        self.toWhisper_path = args.toWhisper_path
        self.temp_file_path = args.temp_file_path
        self.content_extractor = args.content_extractor
        self.random_param_in_toWhisper = args.random_param_in_toWhisper
        self.use_whisper_mix_with_normal = args.use_whisper_mix_with_normal

        # 配置噪声和说话人使用
        self.use_source_noise = args.use_source_noise
        self.use_ref_noise = args.use_ref_noise
        self.use_speaker = args.use_speaker
 
        print(f"use_source_noise: {self.use_source_noise}")
        print(f"use_ref_noise: {self.use_ref_noise}")
        print(f"use_speaker: {self.use_speaker}")
    
        # number of workers
        print(f"Using {NUM_WORKERS} workers")
        self.directory_list = directory_list
        print(f"Loading {len(directory_list)} directories: {directory_list}")

        # Load metadata cache
        # metadata_cache: {file_path: num_frames}
        self.metadata_cache_path = '/mntnfs/lee_data1/caijunwang/ckpt/w2s/rp_metadata_cache.json'
        print(f"Loading metadata_cache from {self.metadata_cache_path}")
        if os.path.exists(self.metadata_cache_path):
            with open(self.metadata_cache_path, 'r', encoding='utf-8') as f:
                self.metadata_cache = json.load(f)
            print(f"Loaded {len(self.metadata_cache)} metadata_cache")
        else:
            print(f"metadata_cache not found, creating new")
            self.metadata_cache = {}

        # Load speaker cache
        # speaker_cache: {file_path: speaker}
        self.speaker_cache_path = '/mntnfs/lee_data1/caijunwang/ckpt/w2s/rp_file2speaker.json'
        print(f"Loading speaker_cache from {self.speaker_cache_path}")
        if os.path.exists(self.speaker_cache_path):
            with open(self.speaker_cache_path, 'r', encoding='utf-8') as f:
                self.speaker_cache = json.load(f)
            print(f"Loaded {len(self.speaker_cache)} speaker_cache")
        else:
            print(f"speaker_cache not found, creating new")
            self.speaker_cache = {}
        
        self.files = []
        # Load all flac files
        for directory in self.directory_list:
            print(f"Loading {directory}")
            files = self.get_flac_files(directory)
            random.shuffle(files)
            self.files.extend(files)
            del files
            print(f"Now {len(self.files)} files.")
            self.meta_data_cache = self.process_files()
            self.speaker_cache = self.process_speakers()
            temp_cache_path = self.metadata_cache_path.replace('.json', f'_{directory.split("/")[-1]}.json')
            if not os.path.exists(temp_cache_path):
                safe_write_to_file(self.meta_data_cache, temp_cache_path)
                print(f"Saved metadata cache to {temp_cache_path}")
            temp_cache_path = self.speaker_cache_path.replace('.json', f'_{directory.split("/")[-1]}.json')
            if not os.path.exists(temp_cache_path):
                safe_write_to_file(self.speaker_cache, temp_cache_path)
                print(f"Saved speaker cache to {temp_cache_path}")
        
        print(f"Loaded {len(self.files)} files")
        random.shuffle(self.files)  # Shuffle the files.

        self.filtered_files, self.all_num_frames, index2numframes, index2speakerid = self.filter_files()
        #只有3-15s的语音才会被保留
        print(f"Loaded {len(self.filtered_files)} files")

        self.index2numframes = index2numframes#index to 每条utt的长度
        self.index2speaker = index2speakerid #index to 每条utt的speaker
        self.speaker2id = self.create_speaker2id() #每条utt的speaker to 每条utt的speaker_id
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )
        del self.meta_data_cache, self.speaker_cache

        if self.use_ref_noise or self.use_source_noise:
            if TRAIN_MODE:
                self.noise_filenames = self.get_all_flac(args.noise_dir)
            else:
                self.noise_filenames = self.get_all_flac(args.test_noise_dir)
                
    def process_files(self):
        print(f"Processing metadata...")
        files_to_process = [file for file in self.files if file not in self.metadata_cache]
        if files_to_process:
            with Pool(processes=NUM_WORKERS) as pool:
                results = list(tqdm(pool.imap_unordered(get_metadata, files_to_process), total=len(files_to_process)))
            for file, num_frames in results:
                self.metadata_cache[file] = num_frames 
            safe_write_to_file(self.metadata_cache, self.metadata_cache_path)
        else:
            print(f"Skipping processing metadata, loaded {len(self.metadata_cache)} files")
        return self.metadata_cache

    def process_speakers(self):
        print(f"Processing speakers...")
        files_to_process = [file for file in self.files if file not in self.speaker_cache]
        if files_to_process:
            with Pool(processes=NUM_WORKERS) as pool:
                results = list(tqdm(pool.imap_unordered(get_speaker, files_to_process), total=len(files_to_process)))
            for file, speaker in results:
                self.speaker_cache[file] = speaker
            safe_write_to_file(self.speaker_cache, self.speaker_cache_path)
        else:
            print(f"Skipping processing speakers, loaded {len(self.speaker_cache)} files")
        return self.speaker_cache

    def get_flac_files(self, directory):
        flac_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".flac") or file.endswith(".wav"):
                    flac_files.append(os.path.join(root, file))
        return flac_files

    def get_all_flac(self, directory):
        directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if not directories:
            return self.get_flac_files(directory)
        with Pool(processes=NUM_WORKERS) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(self.get_flac_files, directories), total=len(directories), desc="Processing"):
                results.extend(result)
        print(f"Found {len(results)} waveform files")
        return results
    
    def get_num_frames(self, index):
        return self.index2numframes[index]
    
    def filter_files(self):
        # Filter files
        metadata_cache = self.meta_data_cache
        speaker_cache = self.speaker_cache
        filtered_files = []
        all_num_frames = []
        index2numframes = {}
        index2speaker = {}
        for file in self.files:
            num_frames = metadata_cache[file]
            if SAMPLE_RATE * 3 <= num_frames <= SAMPLE_RATE * 30:
                filtered_files.append(file)
                all_num_frames.append(num_frames)
                index2speaker[len(filtered_files) - 1] = speaker_cache[file]
                index2numframes[len(filtered_files) - 1] = num_frames
        return filtered_files, all_num_frames, index2numframes, index2speaker
    
    def create_speaker2id(self):
        speaker2id = {}
        unique_id = 0  # 开始的唯一 ID
        print(f"Creating speaker2id from {len(self.index2speaker)} utterences")
        for _, speaker in tqdm(self.index2speaker.items()):
            if speaker not in speaker2id:
                speaker2id[speaker] = unique_id
                unique_id += 1  # 为下一个唯一 speaker 增加 ID
        print(f"Created speaker2id with {len(speaker2id)} speakers")
        return speaker2id
    
    def snr_mixer(self, clean, noise, snr):
        # Normalizing to -25 dB FS
        rmsclean = (clean**2).mean()**0.5
        epsilon = 1e-10
        rmsclean = max(rmsclean, epsilon)
        scalarclean = 10 ** (-25 / 20) / rmsclean
        clean = clean * scalarclean

        rmsnoise = (noise**2).mean()**0.5
        scalarnoise = 10 ** (-25 / 20) /rmsnoise
        noise = noise * scalarnoise
        rmsnoise = (noise**2).mean()**0.5
        
        # Set the noise level for a given SNR
        noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        noisyspeech_tensor = torch.tensor(noisyspeech, dtype=torch.float32)
        return noisyspeech_tensor
    
    def add_noise(self, clean):
        # self.noise_filenames: list of noise files
        random_idx = np.random.randint(0, np.size(self.noise_filenames))
        noise, _ = librosa.load(self.noise_filenames[random_idx], sr=SAMPLE_RATE)
        clean = clean.cpu().numpy()
        if len(noise)>=len(clean):
            noise = noise[0:len(clean)] #截取噪声的长度
        else:
            while len(noise)<=len(clean): #如果噪声的长度小于语音的长度
                random_idx = (random_idx + 1)%len(self.noise_filenames) #随机读一个噪声
                newnoise, fs = librosa.load(self.noise_filenames[random_idx], sr=SAMPLE_RATE)
                noiseconcat = np.append(noise, np.zeros(int(fs * 0.2)))#在噪声后面加上0.2静音
                noise = np.append(noiseconcat, newnoise)#拼接噪声
        noise = noise[0:len(clean)] #截取噪声的长度
        #随机sample一个小于20大于0的随机数
        snr = random.uniform(0.0,20.0)
        noisyspeech = self.snr_mixer(clean=clean, noise=noise, snr=snr) #根据随机的SNR级别，混合生成带噪音频
        del noise
        return noisyspeech
    
    def add_reverb(self, speech):
        room_dim = [np.random.uniform(1, 12) for _ in range(3)]  # [length, width, height]
        mic_pos = [np.random.uniform(0, dim) for dim in room_dim] # 随机选择麦克风位置
        distance = np.random.normal(2, 4) # 确定声源与麦克风的距离
        while distance <= 0 or distance > 5:
            distance = np.random.normal(2, 4)
        source_pos = [mic_pos[0] + distance, mic_pos[1], mic_pos[2]] # 随机选择声源位置，确保它在以麦克风为中心的球内
        rt60 = np.random.uniform(0.05, 1.0) # 随机选择RT60值
        try: 
            rir_filter = rir.generate(
                c=340,                  # 声速
                fs=SAMPLE_RATE,
                r=[mic_pos],            # 麦克风位置
                s=source_pos,           # 声源位置
                L=room_dim,             # 房间尺寸
                reverberation_time=rt60,# RT60值
                nsample=4096,           # IR长度
            )
            # 应用混响
            speech_reverb = np.convolve(speech.cpu().numpy(), rir_filter[:, 0], mode='same')
            speech = torch.tensor(speech_reverb, dtype=torch.float32)
            return speech
        except:
            return speech #如果遇到ValueError: s is outside the room，直接返回没加混响的声音

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.filtered_files[idx]
        speech, wspeech = get_normal_and_whisper(file_path,
                                                 self.temp_file_path,
                                                 self.random_param_in_toWhisper)
                                                #  self.toWhisper_path)
                                                #  "/mntnfs/lee_data1/caijunwang/ckpt/temp",
                                                #  "/mntnfs/lee_data1/caijunwang/lib/toWhisper")
        # import soundfile as sf
        #sf.write("/mntnfs/lee_data1/caijunwang/resources/output.wav", speech, 16000)
        #assert 0
        if len(speech) > 30 * SAMPLE_RATE:
            speech = speech[:30 * SAMPLE_RATE]
            wspeech = wspeech[:30 * SAMPLE_RATE]
        wspeech = torch.tensor(wspeech, dtype=torch.float32)
        speech = torch.tensor(speech, dtype=torch.float32)
        # inputs = torch.tensor(wspeech, dtype=torch.float32)
        if self.content_extractor == "whubert":
            hop_length=320
        elif self.content_extractor == "mhubert":
            hop_length=200
        inputs = self._get_reference_vc(wspeech, speech, hop_length=hop_length)
        speaker = self.index2speaker[idx]
        speaker_id = self.speaker2id[speaker]
        inputs["speaker_id"] = speaker_id
        return inputs
    
    def _get_reference_vc(self, wspeech, speech, hop_length):
        pad_size = 1600 - speech.shape[0] % 1600
        speech = torch.nn.functional.pad(speech, (0, pad_size)) # 保证语音长度是1600的倍数
        wspeech = torch.nn.functional.pad(wspeech, (0, pad_size))

        #hop_size
        frame_nums = speech.shape[0] // hop_length
        clip_frame_nums = np.random.randint(int(frame_nums * 0.25), int(frame_nums * 0.45))
        clip_frame_nums += (frame_nums - clip_frame_nums) % 8
        start_frames, end_frames = 0, clip_frame_nums

        ref_speech = speech[start_frames * hop_length : end_frames * hop_length]
        new_speech = torch.cat((wspeech[:start_frames * hop_length], wspeech[end_frames * hop_length:]), 0)
        tar_speech = torch.cat((speech[:start_frames * hop_length], speech[end_frames * hop_length:]), 0)

        ref_mask = torch.ones(len(ref_speech) // hop_length)
        mask = torch.ones(len(new_speech) // hop_length)

        if self.use_whisper_mix_with_normal:
            mix_speech = mix_audios(new_speech, tar_speech)
        else:
            mix_speech = None

        if not (self.use_source_noise or self.use_ref_noise):
            # 不使用噪声
            return {"speech": new_speech, "ref_speech": ref_speech, "ref_mask": ref_mask, "mask": mask, "target": tar_speech, "mix_speech": mix_speech}
        elif self.use_source_noise and self.use_ref_noise:
            # 使用噪声
            noisy_ref_speech = self.add_noise(ref_speech) # 添加噪声
            nosiy_speech = self.add_noise(new_speech) # 添加噪声
            return {"speech": new_speech, "noisy_speech":nosiy_speech, "ref_speech": ref_speech, "noisy_ref_speech": noisy_ref_speech, "ref_mask": ref_mask, "mask": mask, "target": tar_speech, "mix_speech": mix_speech}
        elif self.use_source_noise and not self.use_ref_noise:
            # 只使用源噪声
            noisy_speech = self.add_noise(new_speech)
            return {"speech": new_speech, "noisy_speech": noisy_speech, "ref_speech": ref_speech, "ref_mask": ref_mask, "mask": mask, "target": tar_speech, "mix_speech": mix_speech}
        elif self.use_ref_noise and not self.use_source_noise:
            # 只使用参考噪声
            noisy_ref_speech = self.add_noise(ref_speech)
            return {"speech": new_speech, "ref_speech": ref_speech, "noisy_ref_speech": noisy_ref_speech, "ref_mask": ref_mask, "mask": mask, "target": tar_speech, "mix_speech": mix_speech}
        
class VCCollator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)
        #self.use_noise = cfg.trans_exp.use_noise

        self.use_source_noise = self.cfg.trans_exp.use_source_noise
        self.use_ref_noise = self.cfg.trans_exp.use_ref_noise
        self.use_whisper_mix_with_normal = self.cfg.trans_exp.use_whisper_mix_with_normal
 
        print(f"use_source_noise: {self.use_source_noise}")
        print(f"use_ref_noise: {self.use_ref_noise}")
 

    def __call__(self, batch):
        packed_batch_features = dict()

        # Function to handle tensor copying
        def process_tensor(data, dtype=torch.float32):
            if isinstance(data, torch.Tensor):
                return data.clone().detach()
            else:
                return torch.tensor(data, dtype=dtype)

        # Process 'speech' data
        speeches = [process_tensor(b['speech']) for b in batch]
        packed_batch_features['speech'] = pad_sequence(speeches, batch_first=True, padding_value=0)

        if self.use_whisper_mix_with_normal:
            # Process 'mix_speech' data
            mix_speeches = [process_tensor(b['mix_speech']) for b in batch]
            packed_batch_features['mix_speech'] = pad_sequence(mix_speeches, batch_first=True, padding_value=0)

        # Process 'ref_speech' data
        ref_speeches = [process_tensor(b['ref_speech']) for b in batch]
        packed_batch_features['ref_speech'] = pad_sequence(ref_speeches, batch_first=True, padding_value=0)

        # Process 'target' data
        target = [process_tensor(b['target']) for b in batch]
        packed_batch_features['target'] = pad_sequence(target, batch_first=True, padding_value=0)

        # Process 'mask' data
        masks = [process_tensor(b['mask']) for b in batch]
        packed_batch_features['mask'] = pad_sequence(masks, batch_first=True, padding_value=0)

        # Process 'ref_mask' data
        ref_masks = [process_tensor(b['ref_mask']) for b in batch]
        packed_batch_features['ref_mask'] = pad_sequence(ref_masks, batch_first=True, padding_value=0)

        # Process 'speaker_id' data
        speaker_ids = [process_tensor(b['speaker_id'], dtype=torch.int64) for b in batch]
        packed_batch_features['speaker_id'] = torch.stack(speaker_ids, dim=0)
        
        if self.use_source_noise:
            # Process 'noisy_speech' data
            noisy_speeches = [process_tensor(b['noisy_speech']) for b in batch]
            packed_batch_features['noisy_speech'] = pad_sequence(noisy_speeches, batch_first=True, padding_value=0)
        if self.use_ref_noise:
            # Process 'noisy_ref_speech' data
            noisy_ref_speeches = [process_tensor(b['noisy_ref_speech']) for b in batch]
            packed_batch_features['noisy_ref_speech'] = pad_sequence(noisy_ref_speeches, batch_first=True, padding_value=0)
        
        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
