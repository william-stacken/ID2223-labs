# ID2223 Lab1
This folder contains two ML projects utilizing Hopsworks, Modal and Huggingface.

## Iris
The first project is able to predict an Iris flower based on its sepal and petal lengths and widths.
See the user interface at Huggingface (https://huggingface.co/spaces/SodraZatre/iris) that allows
the features of the iris flower to be input and the predicted flower to be output.

The project also has a monitoring view (https://huggingface.co/spaces/SodraZatre/iris-monitoring) that
displays the latest flowers in the Hopsworks feature group and the flowers that Modal predicted they were
based on their features. The monitoring view also has a confusion matrix that visualizes the accuracy of the model,
where each box represents the number of times a flower was predcited correctly or incorrectly.

## Titanic