import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'StudyHours': [1,2,3,4,5,6,7,8,9,10],
    'Attendance': [50,55,60,65,70,75,80,85,90,95],
    'Marks': [35,40,45,50,55,60,70,75,85,90],
    'Projects': [1,1,2,2,3,3,4,4,5,5],
    'Pass': [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance', 'Marks']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(pred)
print(accuracy_score(y_test, pred))