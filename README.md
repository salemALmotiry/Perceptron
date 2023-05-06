Before running the perceptron model, ensure that both the train and test files are in the same folder or specify the correct file path.

To initialize the perceptron model with 4 features (x1, x2, x3, x4), run the following:

`p = perceptron(4)`

To filter the data, use the dataFilter() function. This function requires the file path, filter number, and type of file. The filter number can be one of the following:\
`
1: Class 1 with class 2\
2: Class 2 with class 3\
3: Class 1 with class 3\
`
For example, to filter the training data with filter number 2, use the following code:

`X_train, y_train = p.dataFilter('train.data', 2, 'train')`

Once the model is initialized and the data is filtered, the perceptron model is ready to run. You can change the filter number as needed.
