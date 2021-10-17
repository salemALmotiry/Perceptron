import numpy as np
import matplotlib.pyplot as plt

class perceptron:
    def __init__(self, features):
        self.features = features 
        self.weights = np.zeros((features), dtype=float) # init weights with zero 
        self.bias = np.zeros(1, dtype=float) # bias with zero 
        self.errors= np.array([])
        self.test_errors = np.array([])
        self.Train_accuracie = np.array([0])
        self.Test_accuracie = np.array([])
        self.Train_F1_score = np.array([])
        self.Train_Precision = np.array([])
        self.Train_Recall = np.array([])
        self.Test_F1_score = np.array([])
        self.Test_Precision = np.array([])
        self.Test_Recall = np.array([])
        
   
    
    def dataFilter(self,filePath,Filter,Type):
        
        F = {1:'class-1',2:'class-2',3:'class-1'}  
        A = {'test':[0,10,20,30],'train':[0,40,80,120] }     
        data = np.genfromtxt(filePath,delimiter=',',dtype=str)
        X_dataset, y_dataset = data[:, :self.features], data[:, self.features]
        
        if Filter == 1 :
                 X_dataset,y_dataset =X_dataset[A[Type][0]:A[Type][2]],y_dataset[A[Type][0]:A[Type][2]]
            
        if Filter ==2 :    
                 X_dataset,y_dataset =np.concatenate((X_dataset[A[Type][1]:A[Type][2]] , X_dataset[A[Type][2]:A[Type][3]]),axis=0),   np.concatenate((y_dataset[A[Type][1]:A[Type][2]],y_dataset[A[Type][2]:A[Type][3]]),axis=0)
        if Filter == 3:
                 X_dataset,y_dataset =np.concatenate((X_dataset[ A[Type][0]:A[Type][1] ]  ,X_dataset[A[Type][2]:A[Type][3]]),axis=0),   np.concatenate((y_dataset[A[Type][0]:A[Type][1]],y_dataset[A[Type][2]:A[Type][3]]),axis=0)
              
        y_dataset = np.where(y_dataset==F[Filter],1,-1)

        X_dataset,y_dataset = X_dataset.astype(float),y_dataset.astype(float)

        indexes = np.arange(y_dataset.shape[0])
        np.random.shuffle(indexes)
        X_dataset, y_dataset = X_dataset[indexes], y_dataset[indexes]
        
        return X_dataset,y_dataset

    def WeightSum(self,xi):
        return  np.dot(self.weights,xi)+self.bias
   
   
    def active(self,a,yi):
        
        return 1 if  ((yi*a)<=0)  else 0
    
    def update(self,xi,yi):
        
        self.weights = np.add(self.weights,yi*xi)
        self.bias[0] = self.bias[0]+yi
  
   
    def PerceptronTrain(self, X_train,y_train,X_test,y_test,itr):       
        for _ in range(1,itr):
            error = 0 
            for xi , yi in zip(X_train,y_train):
             
                    if self.active(self.WeightSum(xi) , yi):
                        self.update(xi,yi)
                        error+=1
                        
                  
            ac = self.accuracies(X_train,y_train)
            self.Train_accuracie =np.append( self.Train_accuracie,ac[0])
            self.Train_Precision = np.append(self.Train_Precision, ac[2]) 
            self.Train_Recall = np.append(self.Train_Recall, ac[3]) 
            self.Train_F1_score = np.append(self.Train_F1_score, ac[4]) 
                   
            self.errors = np.append(self.errors, error)
            
            ac  = self.accuracies(X_test,y_test)   
            self.Test_accuracie =np.append( self.Test_accuracie,ac[0])
            self.Test_Precision = np.append(self.Test_Precision, ac[2]) 
            self.Test_Recall = np.append(self.Test_Recall, ac[3]) 
            self.Test_F1_score = np.append(self.Test_F1_score, ac[4]) 
            self.test_errors = np.append(self.test_errors, ac[5]) 
            

        
        return self.weights,self.bias   
    
    
    def PerceptronTest(self,x):
        return np.sign( self.WeightSum(x))
        
    
    def accuracies(self,x,y):
        # base on the perceptron test function we make a guess then check if it correct or not 
        # using same as updating part in train function if (weighted_sum*yi)<=0 is true thans means incorrect(wrong guss) 
        # counting the number of correct/incorrect guess to calculate the accuracy and error rate
        
        correct = 0 
        incorrect =  0
        True_Positive = 0 
        False_Positive  = 0 
        false_negative = 0 
        for xi,yi in zip(x,y):
               Predicted  = self.PerceptronTest(xi)
            
               
               if (yi==1):
                  
                   t =  1 if Predicted==yi else 0
                   True_Positive +=t
                   
                   t1 = 1 if Predicted==-1 else 0 
                
                   false_negative +=t1
                
               elif(yi==-1):
               
                   t2 = 1 if Predicted==1 else 0 
                 
                   False_Positive +=t2

               
               if (Predicted *yi>0):
                   
                   correct+=1              
               else :
                   incorrect+=1
               
        total = x.shape[0]
        accuracie = (correct/total)*100.0
        error = 100.0-accuracie
        try :  
         Precision =( True_Positive / (True_Positive + False_Positive ))
        except : 
            Precision = 0 
        try :   
         Recall = ( True_Positive/ (True_Positive + false_negative) )
        except : 
            Recall = 0  
        try : 
         F1_score = 2 * (Precision * Recall) / (Precision + Recall)
        except : 
            F1_score = 0
            
        return accuracie,error,Precision,Recall,F1_score,incorrect
    
    def make_prediction(self,x,y):
        
        for xi,yi in zip(x,y):
            pred = self.PerceptronTest(xi)
            print('X =>',xi,' | ', 'label : ' , '==> ',yi,' | prediction : ', pred , '' if yi==pred else '==> Misprediction')
            
    
    def plot_data(self):
            font1 = {'family':'serif','color':'black','size':20}
            
            plt5 = plt.figure(5)

            plt.plot(self.Train_Precision,c='black', label='Train_Precision')
            plt.plot(self.Test_Precision,c='g', label='Test_Precision' ,linestyle='--')
            plt.title('Precision',fontdict = font1)
            plt.xlabel('Iterations')
            plt.ylabel('Precision')
            plt.legend()
            
            plt4 = plt.figure(4)

            plt.plot(self.Train_Recall,c='black', label='Train_Recall')
            plt.plot(self.Test_Recall,c='g', label='Test_Recall' ,linestyle='--')
            plt.title('Recall',fontdict = font1)
            plt.xlabel('Iterations')
            plt.ylabel('Recall')
            plt.legend()
            
            
            
            plt3 = plt.figure(3)

            plt.plot(self.Train_F1_score,c='black', label='Train_F1_score')
            plt.plot(self.Test_F1_score,c='g', label='Test_F1_score' ,linestyle='--')
            plt.title('F1 score',fontdict = font1)
            plt.xlabel('Iterations')
            plt.ylabel('F1 score')
            plt.legend()

            plt2 = plt.figure(2)

            plt.plot(self.Train_accuracie,c='black', label='Train_accuracie')
            plt.plot(self.Test_accuracie,c='g', label='Test_accuracie' ,linestyle='--')
            plt.title('Accuracies',fontdict = font1)
            plt.xlabel('Iterations')
            plt.ylabel('Accuracie')
            plt.legend()
            
            plt1 = plt.figure(1)
            plt.plot(self.errors,c='black', label='train_Errors')
            plt.plot(self.test_errors,c='g',label='test_test')

            plt.title('Errors',fontdict = font1)
            plt.xlabel('Iterations')
            plt.ylabel('Errors')
            plt.legend()
            
            plt.show()



def main():
    p = perceptron(4)# 4 is the number of features (x1,x2,x3,x4)
    
    # data filter nedd the file path , filter num and type of th file 
    # 1 = class 1 with class 2 
    # 2 = class 2 with class 3 
    # 3 = class 1 with class 3 
    # 
    X_train , y_train = p.dataFilter('train.data',2,'train')       

    X_test,y_test = p.dataFilter('test.data',2,'test')
        
        
    w , b = p.PerceptronTrain(X_train,y_train,X_test,y_test,30)
   
    print('Result :')
    print( '\nweights : ' , p.weights,'bias :', p.bias )
    pp = p.accuracies(X_test,y_test)
    print('Testing : \naccuracie : ', pp[0] ,'%',' , Precision : ', pp[2],', Recall : ',pp[3],', F1_score : ', pp[4])
    print(pp)
    p.plot_data()


if __name__=="__main__":
    main()
    
    
