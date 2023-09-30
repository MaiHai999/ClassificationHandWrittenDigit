import numpy as np

class PCA:
    def __init__(self , n_components):
        self.n_components = n_components

    def fit_transform(self , X):
        X = np.squeeze(X, axis=1)

        self.anpha = np.mean(X, axis=0)
        X_meaned = X - self.anpha
        cov_mat = np.cov(X_meaned, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        self.eigenvector_subset = sorted_eigenvectors[:, 0: self.n_components]
        X_reduced = np.dot(self.eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

        return np.reshape(X_reduced, (X_reduced.shape[0], 1, X_reduced.shape[1]))

    def transform(self , x):
        X_meaned = x - self.anpha
        X_reduced = np.dot(self.eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
        return X_reduced


if __name__ == "__main__":
    x = np.random.rand(20, 20)
    pca = PCA(10)
    Xr = pca.fit_transform(x)


