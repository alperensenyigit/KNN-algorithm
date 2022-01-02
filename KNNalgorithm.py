# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:21:36 2021

@author: Alperen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Outcome = 1 Diabets
# Outcome = 0 Healty
data = pd.read_csv("diabetes.csv")
data.head()

diabets = data[data.Outcome == 1]
healty_people = data[data.Outcome == 0]

# For now, let's make an example drawing just by looking at glucose
plt.scatter(healty_people.Age, healty_people.Glucose, color="green", label="Healty", alpha = 0.4)
plt.scatter(diabets.Age, diabets.Glucose, color="red", label="Diabet", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

# Let's determine x and y axes
y = data.Outcome.values
x_raw_data = data.drop(["Outcome"],axis=1)   
# Remove the outcome column -dependent variable- and leave only independent variables
# Because the KNN algorithm will work within x values..


# Normalize x raw data measures, so they are only between 0 and 1
# If we don't normalize in this way, high numbers will overwhelm 
#   the small numbers and may confuse the KNN algorithm!
x = (x_raw_data - np.min(x_raw_data))/(np.max(x_raw_data)-np.min(x_raw_data))


print("Raw data before normalization:\n")
print(x_raw_data.head())


# sonra 
print("The data we will give for training to AI after normalization:\n\n\n\n")
print(x.head())
    
# We separate our train data and test data
# Our train data will be used to learn the system to distinguish between
#    a healthy person and a sick person.
# And our test data will be used to test whether our machine learning model
#   can accurately distinguish between diabet and healthy people.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("For K=3 our score: %",(knn.score(x_test, y_test))*100)

"""
# What is the best 'k' value ?
count = 1
for k in range(1,11):
    knn_new = KNeighborsClassifier(n_neighbors = k)
    knn_new.fit(x_train,y_train)
    print(count, "  ", "Accuracy: %", knn_new.score(x_test,y_test)*100)
    count += 1
    """
    



