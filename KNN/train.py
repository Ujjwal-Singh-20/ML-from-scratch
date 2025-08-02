import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from KNN import KNN

cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c'])  # blue, orange, green

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=142)

# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors="k", s=20)
# plt.show()

clf = KNN(k=6)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# print(np.unique(y))
print(preds)      #returns most_common[0][0]

acc = np.sum(preds == y_test) / len(y_test)     #34 / 38
print(acc)