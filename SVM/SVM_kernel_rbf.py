import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#let's create the dataset
dataArr=loadmat('../Data/data_train.mat')
labelArr=loadmat('../Data/label_train.mat')
data_name=list(dataArr.keys())[-1]
label_name=list(labelArr.keys())[-1]
dataArr=dataArr[data_name]
labelArr=labelArr[label_name]
dataArr=np.array(dataArr)
labelArr=np.array(labelArr)
X=dataArr
y=labelArr
scaling = StandardScaler()
model = SVC(C=1.0, kernel = 'rbf', gamma=0.1)
model.fit(X,y)
y_pred = model.predict(X)
for i in range(len(y)):
    print("test:",y[i],"prediction:",y_pred[i])
    print("\n")
testArr=loadmat('../Data/data_test.mat')
test_name=list(testArr.keys())[-1]
testArr=testArr[test_name]
testArr=np.array(testArr)
t_pred=model.predict(testArr)
accuracy = accuracy_score(y,y_pred)
print('Total accuracy is:',accuracy)
print(t_pred)
