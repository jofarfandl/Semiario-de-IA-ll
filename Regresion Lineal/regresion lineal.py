import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('countries.csv')
data_mex = data[data.country == "Mexico"]
#data_mex.plot.scatter(x="year",y="lifeExp"

x = np.asanyarray(data_mex[['year']])
y = np.asanyarray(data_mex[['lifeExp']])

model = linear_model.LinearRegression()
model.fit(x,y)
ypred = model.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred,'--r')

from sklearn.metrics import r2_score
print(r2_score(y,ypred))