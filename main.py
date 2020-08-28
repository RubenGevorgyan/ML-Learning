import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import  DecisionTreeClassifier
import  random as rand

#csv files with data are from here https://www.kaggle.com/c/digit-recognizer/data
data = pd.read_csv('data/train.csv').values

classifier = DecisionTreeClassifier()
# our data for training
train_x= data[0:21000,1:]
#our label for training
label = data[0:21000,0]
#training
classifier.fit(train_x,label)


#data vor test
test_x = data[21000:,1:]
#labels for test
actual_label = data[21000: 0]

prediction =classifier.predict(test_x)

counter= 0
for i in range (0, test_x.shape[0]):
    if(prediction[i] ==actual_label[i]):
        counter+= 1
print("Accuracy is" + ((counter/21000)*100))