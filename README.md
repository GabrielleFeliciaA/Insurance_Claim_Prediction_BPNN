# Insurance Claim Prediction Using BackPropagation Neural Networks

## Introduction
Predicting insurance claims using the BackPropagation Neural Network (BPNN) algorithm with Tensorflow Keras. The data used has several variables that will be used to predict whether someone will make an insurance claim or not, these variables include: `age`, `sex`, `bmi`, `steps` (average walking steps per day of the policyholder) , `children`, `smoker`, `region`, and `charges`.

## Exploratory Data Analysis & Data Processing
In this stage, several things are done, namely:
1. Check for missing values ​​in the data and see the statistical distribution of the data. It is known that this data has no missing values.
2. Seeing the correlation between variables. It is known that the `bmi` column and the `steps` column have a high negative correlation, which refers to multicollinearity, and the `steps` column was decided to be deleted because the `bmi` column plays a more important role in predicting whether someone will make an insurance claim or not (People with a high BMI are at risk of experiencing several health conditions related to obesity, thus influencing a person's decision to make an insurance claim).
3. Normalize all independent variables using Z-score normalization.
4. Explore all variables, both numerical and categorical variables in visualization form.
5. Separating the data into 3 parts, namely into a training set of 80%, a validation set of 10% and a testing set of 10%.

## Modelling
At the modeling stage, the first model (Model 1) was created using the backpropagation neural networks algorithm with model 1 architecture having 1 input layer with 7 nodes. Then it has 2 hidden layers with 14 nodes each and 1 output layer with 2 nodes. Model 1 uses the ReLu activation function on each layer.

## Hyperparameter Tuning
The tuning process is carried out using Hyperband to find the best hyperparameters. As a result, the tuned model has the following architecture:
- 1 input layer with 7 nodes.
- Hidden layer 1 with 14 nodes
- Hidden layer 2 with 36 nodes
- 1 Output layer with 2 nodes with activation function softmax
- Learning rate = 0.08
- Number of epochs = 32

## Model Evaluation
Model 1 has the following evaluation results:
- Accuracy : 61%
- Precision value (weighted avg) : 62%
- Recall value (weighted avg) : 61%
- F1-score (weighted avg) : 61%

Meanwhile, the tuned model has the following evaluation results:
- Accuracy : 93%
- Precision value (weighted avg) : 94%
- Recall value (weighted avg) : 93%
- F1-score value (weighted avg) : 93%

From the evaluation results above, it can be seen that the tuned model is better than model 1 because it has higher precision and f1-score values ​​than the model 1. Apart from that, the accuracy and recall values ​​of the tuned model are also higher compared to the model 1.


