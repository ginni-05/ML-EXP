import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = {
    'StudyHours': [1,2,3,4,5,6,7,8,9,10],
    'Attendance': [50,55,60,65,70,75,80,85,90,95],
    'Marks': [35,40,45,50,55,60,70,75,85,90],
    'Pass': [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance', 'Marks']]
y = df['Pass']

lda = LinearDiscriminantAnalysis(n_components=1)
result = lda.fit_transform(X, y)

print(result)