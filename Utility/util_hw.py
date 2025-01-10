import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_iris

def load_toy(n, k, d=2, scale=1, width=0.05, random_state=None):
    '''
    Generates N points sampled from K clusters in R^d space
    '''
    if random_state:
        np.random.seed(random_state)
    if k<0:
        k=5
    centroids = np.random.rand(k, d) * scale
    sizes = divvy(n, k)
    data = np.concatenate([np.c_[np.random.normal(centroid, width, size=(size, d)), np.ones(size) * i]
                           for i, (centroid, size) in enumerate(zip(centroids, sizes))])

    return data[:, :d], data[:, d]

def configure_plots():
    '''Configures plots by making some quality of life adjustments'''
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2

def sample_centroids(X, k, random_state=None):
    '''Sample and return K data points from X'''

    if random_state:
        np.random.seed(random_state)

    indicees = np.random.permutation(np.arange(X.shape[0]))
    centroids = X[indicees[0:k], :]

    assert isinstance(centroids, np.ndarray), 'Your centroids should be in a NumPy array'
    assert centroids.shape == (k, X.shape[1]), f'Your centroids should have shape ({k}, {X.shape[1]})'

    return centroids


def squared_distance(a, b):
    '''A measures a distance between point(s) a and b.'''

    return np.square(np.linalg.norm(a - b, axis=int(len(a.shape) > 1)))


def euclidean(a, b):
    '''Computes the Euclidean distance between point(s) A and another point B'''

    distance = np.linalg.norm(a - b, axis=int(len(a.shape) > 1))

    assert isinstance(distance, (float, np.float64, np.ndarray)), 'Distance should be a float or a NumPy array'
    assert True if not isinstance(distance, np.ndarray) else distance.shape[0] == a.shape[0], \
        'Should have the same number of distances as points in A'

    return distance



def assign(x, centroids, distance_measure=euclidean):
    '''
    Computes the cluster assignments for X or each point
    in X given some centroids and a distance measure
    '''

    assignments = np.c_[[distance_measure(x, centroid) for centroid in centroids]].argmin(axis=0)

    assert np.all(assignments >= 0) and np.all(assignments < len(centroids)), \
        'Assignments should be indices of centroids'

    return assignments



def compute_centroids(X, assignments):
    '''Computes new centroids given points X and cluster ASSIGNMENTS'''

    centroids = np.array([X[assignments == cluster].mean(axis=0)
                     for cluster in np.unique(assignments)])

    assert len(np.unique(assignments)) == len(centroids), \
        'You should have the same number of centroids as clusters'

    return centroids



def fit(X, k, max_iters=1000, tol=1e-2, initial=None, random_state=None):
    '''
    Runs k-means cycle with data X and K clusters
    '''

    if initial is None:
        centroids = sample_centroids(X, k, random_state=random_state)
    else:
        centroids = initial

    assert k == centroids.shape[0]

    for iteration in range(max_iters):
        assignments = assign(X, centroids)
        prev, centroids = centroids, compute_centroids(X, assignments)

        if np.all(np.abs(prev - centroids) < tol):
            break

    return centroids, assignments;


def plot_kmeans(X, centroids, prev_centroids=None, assignments=None):
    '''
    Creates k-means plots
    '''

    # BEGIN SOLUTION

    plt.scatter(X[:, 0], X[:, 1], c=assignments, alpha=0.9, label='data points')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='orange', marker='s', label='final centroids')

    if prev_centroids is not None:
        plt.scatter(prev_centroids[:, 0], prev_centroids[:, 1], s=120,
                    c='gray', marker='s', alpha=0.95, label = 'initial centroids')
        plt.legend()

    plt.title('Toy Clustering Data')
    plt.xlabel('x1')
    plt.ylabel('x2')

    # END SOLUTION

def divvy(total, num_slices):
    '''Divvies TOTAL into NUM_SLICES sizes'''

    size, left = total // num_slices, total % num_slices
    return [size + (left - i > 0) for i in range(num_slices)]
