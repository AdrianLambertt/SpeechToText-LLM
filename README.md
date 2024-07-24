# SpeechToText-LLM
## A light speech-to-text model using open source data for training.  

This project follows in the footsteps of "The AI Hacker's" [youtube video] (https://www.youtube.com/watch?v=YereI6Gn3bM)  

This project has been coded by hand by following the above video, and the associated [github](https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant).  

Many changes have been made between the projects, however the basis of the project is the same.  
The open source voice database used: [Common voice](https://commonvoice.mozilla.org/en)  

### Setup:
As a prequisite to running the network you will need to first run the 'setup_Scripts/mozillaVoice_create_jsons.py' to change the data to the correct format. The Mimic Recording Studio can also be used to create own generated audio files for better training, if this is used you will need to also run 'mimicRecording_create_jsons.py'.  
The project has been designed to run as a Docker container, with the entrypoint for running the network is 'train.py'. This file will require arguments, such as 'python3 train.py --train_file /path/to/folder --valid_file /path/to/folder --root_dir /path/to/folder'  


### Libraries used:
+ Pytorch
+ Lightning
+ Pandas
