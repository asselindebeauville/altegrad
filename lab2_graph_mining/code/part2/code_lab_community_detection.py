"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from random import randint
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    A = nx.adjacency_matrix(G)

    D_inv = diags([1 / G.degree(node) for node in G.nodes()])
    L_rw = eye(G.number_of_nodes()) - D_inv @ A

    eigenvalues, eigenvectors = eigs(L_rw, k=k, which="SR")
    U = eigenvectors.real
    U = normalize(U, axis=1) # Normalizing rows greatly improves modularity

    kmeans = KMeans(n_clusters=k).fit(U)
    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]

    return clustering

############## Task 7

try:
    G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")
except FileNotFoundError:
    G = nx.read_edgelist("../datasets/CA-HepTh.txt", comments="#", delimiter="\t")

largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)
k = 50
clustering = spectral_clustering(subG, k)

############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):

    clusters = set(clustering.values())

    modularity = 0
    m = G.number_of_edges()
    for cluster in clusters:
        nodes_in_cluster = [node for node, cluster_id in clustering.items() if cluster_id == cluster]
        c = G.subgraph(nodes_in_cluster)

        l_c = c.number_of_edges()
        d_c = sum(dict(c.degree()).values())

        modularity += l_c / m - (d_c / (2 * m))**2

    return modularity

############## Task 9

print(f"Modularity for Spectral Clustering: {modularity(subG, clustering):.2f}")

random_clustering = {node: randint(0, k - 1) for node in subG.nodes()}
print(f"Modularity for Random Clustering: {modularity(subG, random_clustering):.2f}")
