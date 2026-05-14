import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Experience': [1,2,3,4,5,6,7,8,9,10],
    'SkillScore': [50,55,60,65,70,75,80,85,90,95],
    'Salary': [25000,30000,35000,40000,45000,50000,60000,65000,70000,80000]
}

df = pd.DataFrame(data)

X = df[['Experience', 'SkillScore']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(pred)
print(mean_squared_error(y_test, pred))