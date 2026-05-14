import pandas as pd
from sklearn.decomposition import PCA

data = {
    'StudyHours': [1,2,3,4,5,6,7,8,9,10],
    'Attendance': [50,55,60,65,70,75,80,85,90,95],
    'Marks': [35,40,45,50,55,60,70,75,85,90],
    'Projects': [1,1,2,2,3,3,4,4,5,5]
}

df = pd.DataFrame(data)

pca = PCA(n_components=2)
result = pca.fit_transform(df)

print(result)