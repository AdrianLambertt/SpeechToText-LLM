import os
import torch
import torch.nn as nn
from model import SpeechRecognition

def train(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)

    model = SpeechRecognition(**h_params)