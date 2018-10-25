import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
dataset = r'C:\Users\Andrew\Desktop\Glass.csv'
data = pd.read_csv(dataset, sep=',')
y = data.Type
x = data.drop('Type', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
classification = KNeighborsClassifier(7)
classification.fit(x_train, y_train)
predictions = classification.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
