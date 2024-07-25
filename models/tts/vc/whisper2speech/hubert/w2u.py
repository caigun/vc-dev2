# code was modified from Wesper: https://github.com/rkmt/wesper-demo
import sys

import numpy as np
import torch
import yaml
import os

import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# HuBERT
from models.tts.vc.whisper2speech.hubert.model import Hubert, URLS, HubertSoft

def wav2units(wav, encoder, layer=None, device='cuda'):
    ''' 
        encoder: HuBERT
    '''
    if type(wav) == np.ndarray:
        wav = torch.tensor([wav], dtype=torch.float32, device=device)
    else:
        wav = wav.to(device)
    assert type(wav) == torch.Tensor
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(0)
    wav = wav.transpose(0, 1)
    # print(wav.shape, "----=-=-=-========-=-=---------")
    #print("#wav2units: ", wav.dtype, wav.shape, min(wav), max(wav))
    with torch.no_grad():  # wav -> HuBERT soft units
        if layer is None or layer < 0:
            #print("#encoder", type(encoder), "device", (next(encoder.parameters())).device)
            #print("#WAV", type(wav), wav.device)
            units = encoder.units(wav) 
        else:
            wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
            units, _ = encoder.encode(wav, layer=layer)
            
    #print("Units", units.shape)
    
    return units


def load_hubert(checkpoint_path=None, rank=0, device='cuda'):
    print("### load_hubert", checkpoint_path, device)
    assert checkpoint_path is not None
    print("### loading checkpoint from: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)
    hubert = HubertSoft().to(device) if device!='cuda' else HubertSoft().to(rank)

    checkpoint = checkpoint['hubert'] if checkpoint['hubert'] is not None else checkpoint
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")

    hubert.load_state_dict(checkpoint, strict=True)
    hubert.eval().to(device)
    return hubert


class whisper2vector(object):
    def __init__(self, cfg, device, load_encoder=True, load_decoder=True, load_vocoder=True, root=None):
        if root is None:
            print("## Lib Local Path", __file__)
            root = os.path.dirname(__file__)
        print("root", root)
        self.root = root

        self.device = device

        self.hubert = cfg.hubert # HuBert
        self.cfg = cfg

        print("MyWhisper2Normal:cfg", cfg)

        # Read Config
        # self.preprocess_config = yaml.load(open(cfg.preprocess_config, "r"), Loader=yaml.FullLoader)
        # self.model_config = yaml.load(open(cfg.model_config, "r"), Loader=yaml.FullLoader)
        # self.configs = (self.preprocess_config, self.model_config)
        
        if load_encoder:
            print("#### loading HuBERT")
            self.encoder = load_hubert(cfg.hubert, device=device) # device 
            print("### HuBERT model", type(self.encoder))

    def forward(self, wav):
        units = wav2units(wav, self.encoder, device=self.device)
        return units
    
    def wav2units(self, wav_t):
        return wav2units(wav_t, self.encoder, device=self.device)
    
