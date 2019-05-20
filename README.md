# EE_with_NN_on_Loan_Data

This is my implementation of entity embedding of categorical data along with the normal numerical data using keras' Functional API.
Data is from https://www.kaggle.com/roshansharma/loan-default-prediction

The purpose of this kernal is predict, given certain features, whether the loan will default or not.

The data is first split into categorical data and numerical data. The numerical data undergoes normalisation and is concatenated with the categorical data to form the featureset. The label is obtained from the original dataframe. The featureset is then vectorised.

The neural network has 27 input nodes, each corresponding to a feature, that are all concatenated together, then introduced into the first FC-layer with 16 neurons. The second FC-layer has 10 neurons. Lastly, the output has a single node with a sigmoid activation function. Binary crossentropy was used as the loss function and gradient descent was done by using RMSprop.
