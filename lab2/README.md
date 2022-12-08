# ID2223 Lab2
This folder contains a feature pipeline, a training pipeline, and an inference pipeline (Huggingface space) for
an ML project that is capable of recognizing swedish speech and output it as text. It utilizes the whisper
pre-trained transformer model for automatic speech recognition.

Original code from [here](https://github.com/ID2223KTH/id2223kth.github.io/blob/master/assignments/lab2/swedish_fine_tune_whisper.ipynb).

## Overview

`whisper-feature-pipeline.py` is a jupyter notebook that fetches Mozilla's common_voice_11_0 dataset in swedish
from Huggingface, spliting it into a train and test set. It then removes all columns from the data except for the
raw audio and the corresponding sentence. After this it preprocesses the audio, including downsampling to 16 kHz
and trancating the length of the audio to 30 seconds. Finally, it prepares the data for the model by computing the
spectogram of the audio and encoding the target sentence to label IDs. Once the features engineering is done,
it saves it to a folder and uploads it as a ZIP archive to Google Drive.

`whisper-training-pipeline.py` is a jupyter notebook that must be executed using Google Colab's GPU runtime.
It first unzips and loads the features and labels produced by the feature pipeline. It then declares a data
collector class that converts the features into pytorch tensors and pads the labels to the maximum length
expected by the model tokenizer. The pipeline then instantiates a trainer from a pre-trained Whisper model
checkpoint, the datacollector, the dataset, and a callback function to calculate the word error rate (WER).
Finally, it starts the training for 4000 training steps, making sure to save checkpoints every 500 steps
to Google Drive to ensure that progress is not lost when Google Colab inevitibly reclaims the used GPU.
Once the training is done, the best-performing model from all checkpoints is pushed to the Huggingface hub.

The gradio UI application can be found in the following [Huggingface space](https://huggingface.co/spaces/SodraZatre/lab2-whisper)
and the model can be found [here](https://huggingface.co/SodraZatre/lab2-whisper-sv).

## Improvement

As can be seen in the huggingface model, it achieves a WER of 19.8946%, which is an improvement over the
[original model](https://github.com/ID2223KTH/id2223kth.github.io/blob/master/assignments/lab2/swedish_fine_tune_whisper.ipynb)
which achieved a WER of 32%. This could be due to using a model-centric-approach and changing the `save_steps` and `eval_steps`
hyperparameters to 500, allowing for more frequent evaluation at the cost of taking longer time to complete. Once the training
finished, it was determined that the model created at step 2500 was the best performing one, and this model would not have been
evaluated without lowering the `eval_steps` parameter.

As for using a data-centric-approach to achieving a better model performance, [this](https://github.com/jim-schwoebel/voice_datasets)
list contains an extensive number of open source voice datasets. For example, [Librispeech](https://www.openslr.org/12) is a dataset
that could compete with Mozilla's common voice dataset. It contains hundreds of hours (several tens of GB) of voice samples.
It is already 16 kHz, so it would not be necessary to downsample it for it to work with Whisper.