# ID2223 Final Project
This folder contains a feature pipeline, a training pipeline, a batch inference pipeline, and two graphical
Huggingface space for predicting the seismic magnitude of a location (given to the model as a latitude and longitude
tuple) and a timestamp (given to the model as unix epoch milliseconds divided by 10^10).

## Overview
`feature-pipeline.py` fetches backfill or realtime data from [USGS's earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)
and stores this data either to a cached CSV file or in the Hopsworks feature group `earthquake_pred`. The features extracted from 
the API include the latitude, longitude, and timestamp of the measurement, and the extracted label is the magnitude. This pipeline
may be configured to run once a day on Modal to continuously fetch updated data from the USGS API.

`training-pipeline.py` creates the Hopsworks feature view `earthquake_pred_view` based on the data in the feature group
`earthquake_pred`. It then splits this data into a train and test set, and feeds the train set into a sequential keras
neural network that evaluates the mean absolute error (MAE) metric of the predicted magnitude labels using the test set.
Finally, the model is uploaded to Hopsworks.

`batch-inference-pipeline.py` fetches the model and the latest data from  `earthquake_pred` and `earthquake_pred_view`,
and uses them to produce comparisons between predictions and the actual labels. These comparisons are uploaded into
the `earthquake_pred_monitoring` feature group and also as a dataframe image in the PNG format. It also produces a heatmap
of evenly spaced points across the valid latitude and longitude input space against the resulting predicted magnitudes.
This is also uploaded to hopsworks as an image in PNG format. The batch inference pipeline should be executed after the
feature pipeline has uploaded new data to `earthquake_pred` and `earthquake_pred_view`.

## Huggingface Spaces
[`hugging-face-spaces-earthquake`](https://huggingface.co/spaces/SodraZatre/earthquake) contains a Gradio interface
that can be used to make predictions based on input features. The user gives a "location" (may be a city, a country, an
address, etc) to the interface and timestamp as a string, where the location is resolved to a latitude and longitude
using the [nominatim](https://nominatim.org/) API.

[`hugging-face-spaces-earthquake-monitoring`](https://huggingface.co/spaces/SodraZatre/earthquake-monitoring) contains a
monitoring Gradio interface that can be fetch the outputs of the batch inference pipeline. This can be used to
infer the performance of the model.
