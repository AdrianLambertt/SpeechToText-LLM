# Script edited from LearnedVector's A-Hackers-Voice-Assistant
# Utility script to convert commonvoice into wav and create the training and test json files from mozilla voice database.

import os
import argparse
import json
import random
import csv
from pydub import AudioSegment

def main(args):
    data = []
    directory = os.path.dirname(args.train_file) # Assumes train & test location are in same dir
    corrupted = 0  # Corrupted files/No path to give an output of total missing/corrupted

    # File opened for debugging / logging output
    with open(args.train_file, encoding='utf-8', errors='ignore') as f:
        train_length = sum(1 for line in f)
        print(str(train_length) + " train files found")
    

    with open(args.train_file, newline='', encoding='utf-8', errors='ignore') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1

        for row in reader:  
            file_name = row['path']
            filename = file_name.rpartition('.')[0] + ".wav"
            text = row['sentence']

            src = directory + "\\clips\\" + file_name
            dst = directory + "\\clips\\" + filename
            
            if(args.logging):
                print("###### Source: " + src + "exists? " + str(os.path.exists(src)) + " Destination: " + dst + "exists? " + str(os.path.exists(dst)))

            
            # If convert and path exists, convert. Else add to corruption. 
            # If not convert, pass & add to data.
            if(args.convert):
                if os.path.exists(src):
                    print("converting file " + str(index) + "/" + str(train_length) + " to wav", end="\r")

                    sound = AudioSegment.from_mp3(src)
                    sound.export(dst, format="wav")
                    index = index + 1

                    data.append({
                    "key": directory +"\\clips\\" + filename,
                    "text": text
                    })
                else:
                    print("###### Source: " + src + "exists? " + str(os.path.exists(src)) + " Destination: " + dst + "exists? " + str(os.path.exists(dst)))
                    corrupted +=1

            else:
                data.append({
                "key": directory + "\\clips\\" + file_name,
                "text": text
                })



    ### TESTING ###


    # File opened for debugging / logging output
    with open(args.test_file, encoding='utf-8', errors='ignore') as f:
        test_length = sum(1 for line in f)
        print(str(test_length) + " test files found")

    with open(args.test_file, newline='', encoding='utf-8', errors='ignore') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1
            
        for row in reader:  
            file_name = row['path']
            filename = file_name.rpartition('.')[0] + ".wav"
            text = row['sentence']


            src = directory + "\\clips\\" + file_name
            dst = directory + "\\clips\\" + filename

            
            if(args.convert):
                if os.path.exists(src):
                    print("converting file " + str(index) + "/" + str(test_length) + " to wav", end="\r")

                    sound = AudioSegment.from_mp3(src)
                    sound.export(dst, format="wav")
                    index = index + 1

                    data.append({
                    "key": directory +"\\clips\\" + filename,
                    "text": text
                    })
                else:
                    print("###### Source: " + src + "exists? " + str(os.path.exists(src)) + " Destination: " + dst + "exists? " + str(os.path.exists(dst)))
                    corrupted +=1

            else:
                data.append({
                "key": directory + "\\clips\\" + file_name,
                "text": text
                })
                
    random.shuffle(data)


    ### JSONs ###


    print("creating JSON's")
    f = open(args.save_json_path +"\\"+ "train.json", "w")
    with open(args.save_json_path +"\\"+ 'train.json','w') as f:
        i=0
        while(i< train_length):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    
    f = open(args.save_json_path +"\\"+ "test.json", "w")
    with open(args.save_json_path +"\\"+ 'test.json','w') as f:
        i= train_length
        while(i< test_length):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    
    print("###### Corrupted/Missing files skipped: " + corrupted)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to convert commonvoice into wav and create the training and test json files for speechrecognition. """
    )
    # Arguments commented due to change in file layout, test and train split has been completed already.

    # parser.add_argument('--file_path', type=str, default=None, required=True,
    #                     help='path to one of the .tsv files found in cv-corpus')
    # parser.add_argument('--percent', type=int, default=10, required=False,
    #                     help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')
    parser.add_argument('--train_file', required=True, help='Path to train.tsv file')
    parser.add_argument('--test_file', required=True, help= 'Path to test.tsv file')
    parser.add_argument('--logging', default=False, help= 'Extra debugging data')
    
    args = parser.parse_args()

    main(args)