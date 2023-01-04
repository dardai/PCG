import faiss
import numpy as np
import time
import torch
import pandas as pd

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def run_kmeans():
    my_matrix = np.loadtxt(open("./bug_vector_results/bug_vectors.csv", "rb"), delimiter=",")
    print(my_matrix)

    x = my_matrix.astype('float32')

    ncentroids = 1000
    niter = 50
    verbose = True
    d = x.shape[1]

    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=False)
    kmeans.train(x)

    cluster_cents = kmeans.centroids
    centroids = torch.Tensor(cluster_cents)
    np.savetxt('./cluster_results/cent.csv', centroids.numpy(), fmt='%.2f', delimiter=',')

    D, I = kmeans.index.search(x, 1)
    insert = np.array([0])
    a = np.insert(I, 0, insert, axis=0)
    data1 = pd.DataFrame(a)
    data1.to_csv('./cluster_results/pre_mapped_index.csv')

    headers = ['bug_id', 'mapped_vector_bugId']

    df = pd.read_csv('./cluster_results/pre_mapped_index.csv', names=headers, skiprows=2)
    df.to_csv('./cluster_results/mapped_index.csv', index=False)
