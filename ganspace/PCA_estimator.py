# Compute PCA on W space

from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import itertools

def get_estimator(name, n_components):
    if name == 'pca':
        return PCAEstimator(n_components)
    if name == 'ipca':
        return IPCAEstimator(n_components)
    else:
        raise RuntimeError('Unknown estimator')

class IPCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(n_components, whiten=self.whiten, batch_size=max(100, 2*n_components))
        self.batch_support = True

    def get_param_str(self):
        return "ipca_c{}{}".format(self.n_components, '_w' if self.whiten else '')

    def fit(self, X):
        self.transformer.fit(X)

    def fit_partial(self, X):
        try:
            self.transformer.partial_fit(X)
            self.transformer.n_samples_seen_ = \
                self.transformer.n_samples_seen_.astype(np.int64) # avoid overflow
            return True
        except ValueError as e:
            print(f'\nIPCA error:', e)
            return False

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_) # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return self.transformer.components_, stdev, var_ratio # PCA outputs are normalized


# Standard PCA
class PCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.solver = 'full'
        self.transformer = PCA(n_components, svd_solver=self.solver)
        self.batch_support = False

    def get_param_str(self):
        return f"pca-{self.solver}_c{self.n_components}"

    def fit(self, X):
        self.transformer.fit(X)

        # Save variance for later
        self.total_var = X.var(axis=0).sum()

        # Compute projected standard deviations
        self.stdev = np.dot(self.transformer.components_, X.T).std(axis=1)

        # Sort components based on explained variance
        idx = np.argsort(self.stdev)[::-1]
        self.stdev = self.stdev[idx]
        self.transformer.components_[:] = self.transformer.components_[idx]

        # Check orthogonality
        dotps = [np.dot(*self.transformer.components_[[i, j]])
            for (i, j) in itertools.combinations(range(self.n_components), 2)]
        if not np.allclose(dotps, 0, atol=1e-4):
            print('IPCA components not orghogonal, max dot', np.abs(dotps).max())

        self.transformer.mean_ = X.mean(axis=0, keepdims=True)

    def get_components(self):
        var_ratio = self.stdev**2 / self.total_var
        return self.transformer.components_, self.stdev, var_ratio