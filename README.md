# Breast-Cancer-Diagnosis-with-ANN
With the Breast Cancer Wisconsin Dataset, an Artificial Neural Network model is created to classify the diagnosis whether the cancer is benign or malignant.

## IDE and Framework
This project is created using Sypder as the main IDE. The main frameworks used in this project are Pandas, Scikit-learn and TensorFlow Keras.

## Methodology
### Data Preprocessing
The datasets are cleaned first by removing unwanted features and label is encoded with label encoder

### Data Pipeline
Data is then split into train-test sets, with a ratio of 70:30.

### Model Pipeline
A feedforward neural network is constructed that is catered for classification problem. The structure of the model is fairly simple with three types of layers:
- Input layer
- Hidden layer
- Output layer

The model is trained with a batch size of 100 and for 100 epochs. The training accuracy of 99% and validation accuracy of 95%. The figure below shows the model accuracy of each epochs
![image](https://user-images.githubusercontent.com/100325884/163185927-f944b511-f148-44c3-b7b7-42498f4b7fb3.png)


## Results
Upon evaluating the model with test data, the model obtain the following test results:
- Test Accuracy: 95%

![image](https://user-images.githubusercontent.com/100325884/163182500-3995d029-215c-4553-95dc-5c3f6dd89974.png)
![image](https://user-images.githubusercontent.com/100325884/163182716-828cb7b1-2098-48c1-acd8-b16442f773ba.png)
