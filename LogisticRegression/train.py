import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LR import LogisticRegression

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

def accuracy(preds, test):
    return np.sum(preds==test)/len(test)

acc = accuracy(y_preds, y_test)
print(acc)