import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Income': [20,22,25,30,80,85,90,95,45,50],
    'Spending': [15,18,20,22,85,88,90,92,50,55]
}

df = pd.DataFrame(data)

model = KMeans(n_clusters=3)
model.fit(df)

df['Cluster'] = model.labels_

print(df)