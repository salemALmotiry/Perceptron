Make sure the train and test files in the same folder or pass the right path 

    p = perceptron(4)# 4 is the number of features (x1,x2,x3,x4)
    
    # data filter need the file path , filter num and type of th file 
    # 1 = class 1 with class 2 
    # 2 = class 2 with class 3 
    # 3 = class 1 with class 3 
    
    X_train , y_train = p.dataFilter('train.data',2,'train') #train.data is the path of file 

The model is ready for run و you can chenge the filter number  
