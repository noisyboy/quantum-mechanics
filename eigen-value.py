import numpy as np
import numpy.linalg as alg

matrix= np.array([[2,4],[6,8]])

eigenval,eigenvec = alg.eigh(matrix)

print(f"Eigen values: {eigenval}")
print(f"Eigen vectors: {eigenvec}")