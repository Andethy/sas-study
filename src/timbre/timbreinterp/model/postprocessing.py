from .. import tools

import torch
from torchaudio.transforms import Spectrogram
from torchaudio.transforms import InverseSpectrogram
import torch.nn as nn
import torch.nn.functional as F

class AudioSpec(nn.Module):
    def __init__(self, audio_configs):
        super().__init__()
        self.spec = Spectrogram(n_fft=audio_configs["n_fft"],
                                win_length=audio_configs["win_length"],
                                hop_length=audio_configs["hop_length"])
        self.fbins = int(audio_configs["n_fft"] / 2)

    def forward(self, x):
        spec_out = self.spec(x)
        spec_out = spec_out[..., :self.fbins, :]

        nframes = spec_out.shape[-1]
        padding = tools.next_power_of_2(nframes) - nframes
        spec_out = F.pad(spec_out, (padding, 0, 0, 0, 0, 0), "constant", 0)

        return spec_out

class Vocoder(nn.Module):
    def __init__(self, audio_configs):
        super().__init__()
        self.inverse_spec = InverseSpectrogram(n_fft=audio_configs["n_fft"],
                                win_length=audio_configs["win_length"],
                                hop_length=audio_configs["hop_length"])
        self.orig_spec_height = audio_configs["orig_spec_height"]
        self.orig_spec_width = audio_configs["orig_spec_width"]

    def forward(self, spec_real, spec_imag):

        spec_real = self.adjust_shape(spec_real)
        spec_imag = self.adjust_shape(spec_imag)
        spec = torch.complex(spec_real, spec_imag)
        audio = self.inverse_spec(spec)

        return audio
    
    def adjust_shape(self, spec):
        spec = F.pad(spec, (0, 0, 0, 1, 0, 0), "constant", 0)
        spec = spec[..., :self.orig_spec_width]

        return spec