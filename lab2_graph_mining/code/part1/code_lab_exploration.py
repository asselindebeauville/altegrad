"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


############## Task 1

try:
    G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")
except FileNotFoundError:
    G = nx.read_edgelist("../datasets/CA-HepTh.txt", comments="#", delimiter="\t")

print(f"The graph has {G.number_of_nodes()} nodes.")
print(f"The graph has {G.number_of_edges()} edges.")

############## Task 2

print(f"The graph has {nx.number_connected_components(G)} connected components.")

largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)

print(f"The largest connected component in the graph has {subG.number_of_nodes()} nodes. ({subG.number_of_nodes() / G.number_of_nodes():.2%})")
print(f"The largest connected component in the graph has {subG.number_of_edges()} edges. ({subG.number_of_edges() / G.number_of_edges():.2%})")

############## Task 3

degree_sequence = [G.degree(node) for node in G.nodes()]

print(f"The minimum degree is {np.min(degree_sequence)}.")
print(f"The maximum degree is {np.max(degree_sequence)}.")
print(f"The median degree is {np.median(degree_sequence)}.")
print(f"The mean degree is {np.mean(degree_sequence)}.")

############## Task 4

histogram = nx.degree_histogram(G)

plt.plot(histogram)
plt.xlabel("frequency")
plt.ylabel("degree")
plt.show()

plt.loglog(histogram)
plt.xlabel("log(frequency)")
plt.ylabel("log(degree)")
plt.show()

############## Task 5

print(f"The global clustering coefficient of the graph is {nx.transitivity(G)}.")
