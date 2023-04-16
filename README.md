# Bird Sound Detection
## Description
In this repo, I build a simple classification network to solve the task of Bird Sound Detection (determine whether bird sound is contained in audio file or not). Additionally, I experiment some audio features on this task to see which one fit well on this challenge. The code is implemented in Tensorflow.

## Dataset
The dataset used in this repo is __Warblr__. You can look for detail information of this dataset and others in this [link](https://dagshub.com/kingabzpro/Bird-Audio-Detection-challenge).

When downloading completed, unzip and put the dataset into `./data`.
## Experimental Results

## Installation
```
$ pip install -r requirements.txt
```
__Note:__ There will be a problem when installing librosa library if you use conda environment, you can fix this by following this [instruction](https://stackoverflow.com/questions/62658071/modulenotfounderror-no-module-named-librosa).
## Instruction
You can test some audio processing function on a single audio file by running:
```
$ python utils/data_processing.py \
    --audio "path/to/audio/file" \
    --features spectrogram \
    --is_visualize True
```
For training, you just simply run:
```
$ python train.py
```