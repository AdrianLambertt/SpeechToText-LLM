import torch
import torch.nn as nn
import torchaudio

class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectogram(
            sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length
        )

    def forward(self, x):
        x = self.transform(x)
        x = torch.log(x + 1e-14) #Compression of values to fit modelling
        return x



class SpecAugment(nn.Module):

    def __init__(self, rate, policy, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

    def forward(self, x):
        return self.specaug(x)