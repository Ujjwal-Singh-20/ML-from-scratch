import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)  #so that exp(x) stays within safe numerical bounds, exp(500) is huge but not undefined like in case of exp(>1000)
    return (1 / (1 + np.exp(-x)))


class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.biases = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.biases
            preds = sigmoid(linear_pred)   #activation function

            #gradients
            dw = (1/n_samples) * np.dot(X.T, (preds-y))
            db = (1/n_samples) * np.sum(preds-y)

            #weight and bias update using the gradients
            self.weights = self.weights - (self.lr * dw)
            self.biases = self.biases - (self.lr * db)


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.biases
        y_pred = sigmoid(linear_pred)

        class_preds = [0 if y<=0.5 else 1 for y in y_pred]

        return class_preds
    
