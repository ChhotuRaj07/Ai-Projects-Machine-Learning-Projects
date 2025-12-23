import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load DataSet

iris = load_iris()

# Features and Labels 

x = iris.data  # sepel & petal data
y = iris.target  # flower type

#Split dataset into training & Testing 

X_train, X_test, y_train, y_test = train_test_split(
     x,y, test_size=0.2, random_state=42
)

#create a ml model 
model = KNeighborsClassifier(n_neighbors=3)

#train your Model 
model.fit(X_train,y_train)

# Make Predictions 
y_pred = model.predict(X_test)

# check Accurucy_score 

accuracy =  accuracy_score(y_test, y_pred)
print("Model Accuracy", accuracy)


sample = [[5.1, 3.5, 1.4, 0.2]]

prediction  = model.predict(sample)

# show flower Name

flower_name  = iris.target_names[prediction]
print("predicted Flower:", flower_name[0]) 

