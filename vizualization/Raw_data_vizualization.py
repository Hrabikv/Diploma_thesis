import plotly.express as px
import os
import numpy as np
import scipy as sc
from sklearn.manifold import TSNE


def plot_data_of_all_subject(data, labels):
    data = np.concatenate(data)
    labels = np.concatenate(labels)

    X = []
    for x in data:
        X.append(np.concatenate(x))

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)

    X_embedded = tsne.fit_transform(np.array(X))
    fig = px.scatter(x=X_embedded.T[0], y=X_embedded.T[1], color=labels)
    fig.show()


def plot_data_of_subject(data, labels):
    X = []
    for x in data:
        X.append(np.concatenate(x))

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)

    X_embedded = tsne.fit_transform(np.array(X))
    fig = px.scatter(x=X_embedded.T[0], y=X_embedded.T[1], color=labels)
    fig.show()
    print()


def plot_data_sample(data_sample):
    print()