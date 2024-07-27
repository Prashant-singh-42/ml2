# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
import pickle

# # Load the csv file
# df = pd.read_csv("iris.csv")

# print(df.head())

# # Select independent and dependent variable
# X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
# y = df["Class"]

# # Split the dataset into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# # Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test= sc.transform(X_test)

# # Instantiate the model
# classifier = RandomForestClassifier()

# # Fit the model
# classifier.fit(X_train, y_train)

# # Make pickle file of our model
# pickle.dump(classifier, open("model.pkl", "wb"))

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

Standard_obj = StandardScaler()
Standard_obj.fit(x_train)
x_train_std = Standard_obj.transform(x_train)
x_test_std = Standard_obj.transform(x_test)

knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
knn.fit(x_train_std,y_train)
# print('Training data accuracy {:.2f}'.format(knn.score(x_train_std, y_train)*100))
# print('Testing data accuracy {:.2f}'.format(knn.score(x_test_std, y_test)*100))
features = np.array([[3.4, 4, 5, 1]])

prediction = knn.predict(features)

print(prediction)

pickle.dump(knn, open("model.pkl", "wb"))

