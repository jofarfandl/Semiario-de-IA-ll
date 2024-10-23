"""
#REGRESION LINEAL MULTIPLE CON COLORES
#independientesdepen,predicc,obj
#x1 x2 x3 x4 . . . xn y
#parametro ->Todo lo que ees aprendible


Explicabilidad  -  poder de prediccion (Tbla)

MSE = 1/M m|sumatoria|i=1 (yi-yestimadai)**2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('home_data.csv')
#Primer paso, limpiar los datos#eleccion de variables
y = np.asanyarray(data[['price']])
x = np.asanyarray(data.drop(columns=['id','price','date']))

#escalamos los datos
scaler = StandardScaler()
x = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest, = train_test_split(x,y,test_size=0.1) 

#Crear y entrenar modelo
model = LinearRegression()
model.fit(xtrain, ytrain)

print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))

#extraer coeficientes NORMLAMENTE SE HACE ESTO PARA EXPLICAR VARIABLES
coef = np.abs(model.coef_.ravel())
df = pd.DataFrame()
names = np.array(data.drop(columns = ['id','price','date']).columns)

df['names'] = names
df['coef'] = coef / np.sum(coef)
df.sort_values(by='coef', ascending=False, inplace = True)
df.set_index('names', inplace = True)
df.coef.plot(kind = 'bar')