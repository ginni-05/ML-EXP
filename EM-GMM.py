import pandas as pd
from sklearn.mixture import GaussianMixture

data = {
    'Income': [20,22,25,30,80,85,90,95,45,50],
    'Spending': [15,18,20,22,85,88,90,92,50,55]
}

df = pd.DataFrame(data)

gmm = GaussianMixture(n_components=3)

gmm.fit(df)

labels = gmm.predict(df)

df['Cluster'] = labels

print(df)