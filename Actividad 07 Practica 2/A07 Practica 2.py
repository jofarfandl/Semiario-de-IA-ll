import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("HumanR_data.csv")

data = data.drop(columns=["Employee_Name","DateofHire","EmploymentStatus","DateofTermination","HispanicLatino","EmpID", "MarriedID","MaritalStatusID", "GenderID","EmpStatusID","DeptID", "PerfScoreID","DOB","MaritalDesc","FromDiversityJobFairID", "Termd","PositionID", "TermReason","ManagerName", "ManagerID","RecruitmentSource","PerformanceScore","EngagementSurvey","LastPerformanceReview_Date","DaysLateLast30"])  # Reemplaza "columna1" y "columna2" con los nombres de las columnas a eliminar.

le = LabelEncoder()  
scaler = StandardScaler()  

data["Position"] = le.fit_transform(data["Position"])
data["Department"] = le.fit_transform(data["Department"])
data["State"] = le.fit_transform(data["State"])
data["Sex"] = le.fit_transform(data["Sex"])
data["CitizenDesc"] = le.fit_transform(data["CitizenDesc"])
data["RaceDesc"] = le.fit_transform(data["RaceDesc"])


X = data.drop(columns=["Salary"])
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")