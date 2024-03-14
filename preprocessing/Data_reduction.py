import plotly.express as px
import os
import numpy as np
import scipy as sc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mne.decoding import CSP
from scipy.stats import lognorm


def _concatenate_data(data):
    X = []
    for x in data:
        X.append(np.concatenate(x))
    return np.array(X)


def apply_TSNE(data):
    X = _concatenate_data(data)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)

    X_embedded = tsne.fit_transform(X)
    return X_embedded


def apply_CSP(data, labels):
    # X = _concatenate_data(data)

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    X_embedded = csp.fit_transform(data, labels)
    return X_embedded


def apply_PCA(data):
    X = _concatenate_data(data)

    pca = PCA(n_components=2, svd_solver='full')

    X_embedded = pca.fit_transform(X)
    return X_embedded


def apply_Lognorm():
    print()