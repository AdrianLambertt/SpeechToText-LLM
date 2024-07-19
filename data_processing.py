import torch
import torch.nn as nn
import torchaudio
from utils import TextProcess
import pandas as pd

class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectogram (
            sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length
        )

    def forward(self, x):
        x = self.transform(x)
        x = torch.log(x + 1e-14) #Compression of values to fit modelling
        return x


# Specaugment applies augmentation techniques to the spectograms.
# The different policies increase the variation of sounds sent to the neural model.
# The variation should lead to a more robust system.
class SpecAugment(nn.Module):

    def __init__(self, rate, policy, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )


    def forward(self, x):
        return self._forward(x)
    
    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x
    
    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x
    
    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


class data(torch.utils.data.Dataset):

    #Preset params that can be overwritten
    parameters = {
        "sample_rate": 8000, "n_feats":81, "specaug_rate": 0.5,
        "time_mask": 70, "freq_mask":15
    }


    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, time_mask, freq_mask, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()

        print("Loading JSON file...")
        self.data = pd.read_json(json_path, lines=True)

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaug_rate, freq_mask, time_mask)
            )

    def __len__(self):
        return len(self.data)
    
    #Get .wav file from dataset.
    #Checks if idx is int, turn to int if not
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int_sequencer(self.data['text'].iloc[idx])
            spectogram = self.audio_transforms(waveform)
            spec_len = spectogram.shape[-1]
            label_len = len(label)
            if spec_len > label_len:
                raise Exception("Spectogram length is larger than label length")
            if spectogram.shape[0] > 1:
                raise Exception("Dual channel, audio file skipped %s"%file_path)
            if spectogram.shape[2] > 1650:
                raise Exception("Spectogram too big. Size: %s" %spectogram.shape[2])
            if label_len == 0:
                raise Exception("Label length is zero. Skipping. %s"%file_path)
            
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx-1 if idx != 0 else idx + 1)
        return spectogram, label, spec_len, label_len
    

    def describe(self):
        return self.data.describe()
    

    #TODO not sure if collate_fn_padd is nessecary? will keep out for now.