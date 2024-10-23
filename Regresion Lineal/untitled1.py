import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearNeuron:
    def __class__init__(self, n_input):
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
    
    def predict(self, X):
        Y_est = np.dot(self.w, X) + self.b
        return Y_est
    
    def batcher(self, X, Y, batch_size):
        p = X.shape[1]
        li, ui = 0, batch_size
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li + batch_size, ui + batch_size
            else:
                return None
    
    def MSE(self, X, Y):
        p = X.shape[1]
        Y_est = self.predict(X)
        return (1/p) * np.sum((Y -Y_est) ** 2)
    
    def fit(self,X, Y, epochs=500, lr = 0.08, batch_size = 16):
        p = X.shape[1]
        error_history = []
        
        for _ in range(epochs):
            miniBatch = self.batcher(X, Y, batch_size)
            for mX, mY in miniBatch:
                mY_est = self.predict(mX)
                self.w += (lr/p) * ((mY-mY_est) @ mX.T).ravel()
                self.b += (lr/p) * np.sum(mY - mY_est)
                error_history.append(self.MSE(X, Y))
        return error_history
    


"""
Ejemplo
p=100
x= -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 + 3 * np.random.rand(p)

                
"""   
data = pd.read_csv("reg_problem.csv")
x = data["x"].values.reshape(1, -1)
y = data["y"].values.reshape(1, -1)    

neurona = LinearNeuron(1)
error = neurona.fit(x, y, batch_size=77)

#Dibujo
plt.plot(x,y, '.b')
xnew = np.array([[-5,5]])
ynew = neurona.predict(xnew)
plt.plot(xnew.ravel(), ynew, '-r')

plt.figure()
plt.plot(error)
plt.show()     








