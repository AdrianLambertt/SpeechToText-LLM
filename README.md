# SpeechToText-LLM
## A light speech-to-text model using open source data for training.  

**This is a learning experience to have a more hands-on experience with coding neural networks in python, instead of MATLAB.**

## Acknowledgements
This project has been heavily inspired by [LearnedVector](https://github.com/LearnedVector) on their [GitGub](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant). Their implementation served as a guide while I was learning Pytorch.  
[Youtube video associated to the above repository](https://www.youtube.com/watch?v=YereI6Gn3bM)

I have adapted and modified parts of the original code to fit my own style and project requirements. Specifically, changes have been made to train.py and data_processing.  

**The open source voice database used: [Common voice](https://commonvoice.mozilla.org/en)**  

## Dependencies
+ Python3
+ [kenlm](https://github.com/kpu/kenlm) language model used to more accurately predict sentances/speech patterns.
+ [ctcdecode](https://github.com/parlance/ctcdecode) Rescoring algorithm using the language model to build outputs (beams) with probabilites of each word. `Red a book: 35%` VS `Read a book: 0.95%`

### Setup
As a prequisite to running the network you will need to first run the `setup_Scripts/mozillaVoice_create_jsons.py` to change the data to the correct format. The Mimic Recording Studio can also be used to create own generated audio files for better training, if this is used you will need to also run `mimicRecording_create_jsons.py`.  
The project has been designed to run as a Docker container, with the entrypoint for running the network is `train.py`. This file will require arguments, such as `python3 train.py --train_file /path/to/folder --valid_file /path/to/folder --root_dir /path/to/folder`


### Libraries used
+ Pytorch
+ Lightning
+ Pandas
