import numpy as np
from sklearn.decomposition import TruncatedSVD

data = [
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
]

X = np.array(data)

svd = TruncatedSVD(n_components=2)

result = svd.fit_transform(X)

print("Reduced Matrix:")
print(result)