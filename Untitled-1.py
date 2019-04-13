import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
iris_data=iris.data
iris_data=pd.DataFrame(iris_data,columns=iris.feature_names)
iris_data.head()
#adding column
iris_data=iris_data.assign(species=iris.target)

X=iris_data.values[:,0:4]
Y=iris_data.values[:,4]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
#model=LogisticRegression()
model=KNeighborsClassifier()
model.fit(x_train,y_train)
species=['setosa','versicolor','virginica']
#predict 
prediction=model.predict(x_test)
#Accuracy
accuracy_score(y_test,prediction.round())