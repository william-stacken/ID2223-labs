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

Original code from https://github.com/ID2223KTH/id2223kth.github.io/tree/master/src/serverless-ml-intro

## Titanic
The second project is similar to the first one, execpt it predicts whether a passenger would have survived
the Titanic disaster based on the features the class, age, sex, deck, family size, and fare per person.
It utilizes the Titanic dataset (https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv)
and performs feature engineering and data cleaning to allow construtcing a model.
See the user interface at Huggingface (https://huggingface.co/spaces/SodraZatre/) that allows
the features of the passenger to be input and the predicted survival or non-survival to be output.

This project also has a monitoring view (https://huggingface.co/spaces/SodraZatre/iris-monitoring) that displays the
last prediction results by Modal and a confusion matrix.
