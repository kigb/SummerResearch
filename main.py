import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
import time

# Define the function for diffusion map
def diffusion_map(X, n_components=2, epsilon='auto'):
    # Compute the pairwise Euclidean distances
    dists = euclidean_distances(X)

    # Compute the kernel matrix
    if epsilon == 'auto':
        epsilon = np.median(dists)
    K = np.exp(-dists**2 / epsilon**2)

    # Construct the Markov matrix
    P = K / K.sum(axis=1, keepdims=True)

    # Compute the eigenvalues and eigenvectors
    lambdas, vectors = eigh(P)

    # Sort by eigenvalues in descending order and select the top 'n_components'
    idx = np.argsort(lambdas)[::-1][:n_components]
    lambdas = lambdas[idx]
    vectors = vectors[:, idx]

    # Compute the diffusion map
    diffusion_map = vectors * lambdas[None, :]

    return diffusion_map



# Function to compare different dimensionality reduction methods
def compare_dimensionality_reduction_methods(X, n_components=2):
    # Define the dimensionality reduction methods to compare
    methods = {
        'Diffusion map': diffusion_map,
        't-SNE': TSNE(n_components=n_components).fit_transform,
        'Isomap': Isomap(n_components=n_components).fit_transform,
        'LLE': LocallyLinearEmbedding(n_components=n_components).fit_transform,
        'MDS': MDS(n_components=n_components).fit_transform,
        'PCA': PCA(n_components=n_components).fit_transform
    }

    # Initialize a dictionary to store the timings
    timings = {}

    # Apply each method to the data and record the time taken
    for method_name, method in methods.items():
        start_time = time.time()
        method(X)
        end_time = time.time()
        timings[method_name] = end_time - start_time

    return timings

# Generate a random dataset
np.random.seed(0)
X = np.random.normal(size=(1000, 20))

# Compare the methods
timings = compare_dimensionality_reduction_methods(X)

# Plot the timing results as a bar chart
plt.barh(list(timings.keys()), list(timings.values()), log=True)  # Use log scale because of large differences in times
plt.xlabel('Time (s, log scale)')
plt.title('Comparison of Dimensionality Reduction Methods')
plt.show()
