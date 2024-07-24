# SpeechToText-LLM
## A light speech-to-text model using open source data for training.  

**This is a learning experience to have a more hands-on experience with coding neural networks in python, instead of MATLAB.**
This project follows in the footsteps of , and the associated [github](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant).  

Many changes have been made between the projects, as the project is four years old, and without explanatory comments.
**The open source voice database used: [Common voice](https://commonvoice.mozilla.org/en)**  

## Youtube video guidance
I Built a Personal Speech Recognition System for my AI Assistant by The AI Hacker: [youtube video](https://www.youtube.com/watch?v=YereI6Gn3bM)

### Setup
As a prequisite to running the network you will need to first run the `setup_Scripts/mozillaVoice_create_jsons.py` to change the data to the correct format. The Mimic Recording Studio can also be used to create own generated audio files for better training, if this is used you will need to also run `mimicRecording_create_jsons.py`.  
The project has been designed to run as a Docker container, with the entrypoint for running the network is `train.py`. This file will require arguments, such as `python3 train.py --train_file /path/to/folder --valid_file /path/to/folder --root_dir /path/to/folder`


### Libraries used
+ Pytorch
+ Lightning
+ Pandas
