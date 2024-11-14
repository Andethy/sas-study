from .. import tools

from torchaudio.transforms import Spectrogram
import torch.nn as nn
import torch.nn.functional as F

class AudioSpec(nn.Module):
    def __init__(self, audio_configs):
        super().__init__()
        self.get_complex_spec = Spectrogram(n_fft=audio_configs["n_fft"],
                                win_length=audio_configs["win_length"],
                                hop_length=audio_configs["hop_length"],
                                power=None)
        self.fbins = int(audio_configs["n_fft"] / 2)

    def forward(self, x):
        spec = self.get_complex_spec(x)
        spec_real = spec.real
        spec_imag = spec.imag

        spec_real = self.adjust_shape(spec_real)
        spec_imag = self.adjust_shape(spec_imag)

        return spec_real, spec_imag
    
    def adjust_shape(self, spec):
        spec = spec[..., :self.fbins, :]
        nframes = spec.shape[-1]
        padding = tools.next_power_of_2(nframes) - nframes
        spec = F.pad(spec, (0, padding, 0, 0, 0, 0), "constant", 0)

        return spec
