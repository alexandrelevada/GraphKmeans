#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to reproduce the second set of experiments described in the paper:

A graph-based k-means algorithm using geodesic distances


"""

# Imports
import sys
import time
import warnings
import networkx as nx
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
import scipy.sparse._csr
from numpy import dot
from numpy import trace
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import v_measure_score


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Topological K-means implementation
def TopKmeans(dados, nn):
    print('Running Topologic K-means...')
    iteracao = 0
    # Number of samples, features and classes
    n = dados.shape[0]
    m = dados.shape[1]
    k = len(np.unique(target))
    # Initial random centers
    coord_centros = np.random.choice(n, size=k, replace=False)    
    centros = dados[coord_centros]
    # Add centers at the end of data matrix (k last rows)
    dados = np.vstack((dados, centros))
    # Main loop
    while True:
        iteracao += 1  
        # Builds the k-NN graph
        knnGraph = sknn.kneighbors_graph(dados, n_neighbors=nn, mode='distance')
        # Adjacency matrix
        W = knnGraph.toarray()
        # NetworkX format
        G = nx.from_numpy_array(W)
        # Assure the k-NN graph is connected
        while not nx.is_connected(G):
            nn += 1
            knnGraph = sknn.kneighbors_graph(dados, n_neighbors=nn, mode='distance')
            # Adjacency matrix
            W = knnGraph.toarray()
            # NetworkX format
            G = nx.from_numpy_array(W)
        # Array for geodesic distances
        distancias = np.zeros((k, n+k))
        for j in range(k):
            # Dijkstra's algorithm for geodesic distances
            length, path = nx.single_source_dijkstra(G, coord_centros[j])
            # Sort vertices
            dists = list(dict(sorted(length.items())).values()) 
            distancias[j, :] = dists
        # Labels vector
        rotulos = np.zeros(n)    
        # Assign labels to data points
        for j in range(n):
            rotulos[j] = distancias[:, j].argmin()
        # Find the points belonging to each partition
        novos_centros = np.zeros((k, m))
        for r in range(k):
            indices = np.where(rotulos==r)[0]
            if len(indices) > 0:
                sample = dados[indices]
                novos_centros[r, :] = sample.mean(axis=0)
            else:
                novos_centros[r, :] = centros[r, :]
        # Update the last k rows of data (centroids)
        dados[n:n+k, :] = novos_centros
        # Check for convergence
        if (np.linalg.norm(centros - novos_centros) < 0.5) or iteracao > 20:
            break
        # Update the centers
        centros = novos_centros.copy()   
    return rotulos

#%%%%%%%%%%%%%%%%%%%%  Data loading
##### Second set of experiments
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='abalone', version=1)
#X = skdata.fetch_openml(name='cnae-9', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='semeion', version=1)
#X = skdata.fetch_openml(name='vehicle', version=1)
#X = skdata.fetch_openml(name='micro-mass', version=1)
#X = skdata.fetch_openml(name='har', version=1)
#X = skdata.fetch_openml(name='waveform-5000', version=1)
#X = skdata.fetch_openml(name='texture', version=1)
#X = skdata.fetch_openml(name='mnist_784', version=1)
#X = skdata.fetch_openml(name='audiology', version=1)
#X = skdata.fetch_openml(name='yeast', version=1)

dados = X['data']
target = X['target']  

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))
nn = round(np.sqrt(n))

print('N = ', n)
print('M = ', m)
print('C = ', c)
print('K = ', nn)
print()
input('Press enter to continue...')
print()

# Sparse matrix (for some high dimensional datasets)
if type(dados) == scipy.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
        target = target.to_numpy()

# Remove nan's
dados = np.nan_to_num(dados)

# Convert labels to integers
rotulos = list(np.unique(target))
numbers = np.zeros(n)
for i in range(n):
    numbers[i] = rotulos.index(target[i])
target = numbers

#########################
# Execution of TopKmeans
#########################
MAX = 31
inicio = time.time()
lista_rand = []
lista_mi = []
lista_v = []
for i in range(1, MAX):
    labels = TopKmeans(dados, nn)
    # External indices
    lista_rand.append(rand_score(target, labels))
    lista_mi.append(mutual_info_score(target, labels))
    lista_v.append(v_measure_score(target, labels))
fim = time.time()

print()
print('GEODESIC K-MEANS')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Average Rand index: %.4f' %(sum(lista_rand)/(MAX-1)))
print('Average Mutual information: %.4f' %(sum(lista_mi)/(MAX-1)))
print('Average V-measure: %.4f' %(sum(lista_v)/(MAX-1)))

########################
# Execution of HDBSCAN 
########################
inicio = time.time()
lista_rand = []
ista_mi = []
lista_v = []
for i in range(1, MAX):
    hdbscan = HDBSCAN(min_cluster_size=10).fit(dados)
    lista_rand.append(rand_score(target, hdbscan.labels_))
    lista_mi.append(mutual_info_score(target, hdbscan.labels_))
    lista_v.append(v_measure_score(target, hdbscan.labels_))
fim = time.time()

print()
print('HDBSCAN')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Average Rand index: %.4f' %(sum(lista_rand)/(MAX-1)))
print('Average Mutual information: %.4f' %(sum(lista_mi)/(MAX-1)))
print('Average V-measure: %.4f' %(sum(lista_v)/(MAX-1)))