# Daniel Tran
#CS445 Machine Learning Program #3

import numpy as np
import random
from matplotlib import pyplot as plt

#given dataset
data_set = './545_cluster_dataset (1).txt'

#load the dataset
def dataload(file_path):
    with open (file_path, 'r') as data:
        data = np.genfromtxt(file_path, delimiter='  ')
    return data

#select k data points using a random permutation
def r_k(k, data):
    k_data = np.random.permutation(data)[:k]
    print(f"initial chosen: \n {k_data}")
    return k_data

#assign each point to closest mean
def c_mean(k, k_data, data):
    ctr = 0
    ptr = 0
    member = np.zeros(k) 
    #euclidiean distance
    distance = np.zeros(k) 
    #clusters assigned for points
    cluster = np.zeros(1500) 
    for row in data:
        for k in k_data:
           distance[ctr] = np.sqrt(np.square(k[0] - row[0]) + np.square(k[1] - row[1]))
           ctr += 1
        ctr = 0
        #ptr of min distance
        minimum = np.argmin(distance) 
        cluster[ptr] = minimum
        member[minimum] += 1
        ptr += 1
    return cluster, member

#re_mean means
def re_mean(k, data, cluster, member):
    add = np.zeros((k,2))
     #new means
    n_kmean = np.zeros((k,2))
    ctr = 0
    for pt in cluster:
        ptr = int(pt)
        add[ptr,0] += data[ctr,0]
        add[ptr,1] += data[ctr,1]
        ctr += 1
    ctr = 0
    for a in add:
        if member[ctr] != 0:
            a = a/member[ctr]
            n_kmean[ctr,0] = a[0]
            n_kmean[ctr,1] = a[1]
        ctr+= 1

    return n_kmean

#K-Means Algorithm
def k_mean(k, k_data, data):
    old_k = np.copy(k_data)
    cluster, member = c_mean(k, k_data, data)
    n_kmean = re_mean(k, data, cluster, member)
    comparison = old_k == n_kmean
    equal = comparison.all()

    if (equal == False):
        # run K-Means
        n_kmean, cluster, E = k_mean(k, n_kmean, data) 
    else:
        print("k_data finished. The final centroids are:")
        print(n_kmean)
        E = sumsquareerror(k, n_kmean, cluster, data)
        print(f"sum square error: {E}")
        plot = input("Enter 1 to print:\n")
        plot = int(plot)
        if plot == 1:
            plotpoints(k, n_kmean, cluster, data)
    return n_kmean, cluster, E

#run K-Means r times 
def run_k_means(data):
    run = 0
    r = input("Please enter the number of times to run K-Means:\n")
    r = int(r)
    k = input("Please enter the number of clusters:\n")
    k = int(k)    
    #SSE 
    lowest_SSE = np.zeros(r)
    #final means
    best_cluster = np.zeros ((r,k,2)) 
    while (r != 0):
        print(f"Run {run + 1}:")
        k_data = r_k(k, data)
        final, cluster, E = k_mean(k, k_data, data)
        lowest_SSE[run] = E
        best_cluster[run] = final
        r -= 1
        run += 1
    #min distance
    minimum = np.argmin(lowest_SSE) 
    print(f"Lowest sum square error was run {minimum + 1} with value {lowest_SSE[minimum]} with clusters:")
    print(f"{best_cluster[minimum]}") 
    
#sum of square error
def sumsquareerror(k, k_data, cluster, data):
    rss = np.zeros(k)
    ctr = 0
    E = 0
    for row in data:
        c = cluster[ctr]
        c = int(c)
        rss[c] += np.square((k_data[c,0] - row[0]) + (k_data[c,1] - row[1]))
        ctr += 1
    for ptr in rss:
        E += ptr
    return E

#plot K-Means clusters
def plotpoints(k, k_data, cluster, data):
    colors = np.zeros((k, 3))
    ctr = 0
    while ctr < k:
        r = random.random()
        b = random.random()
        g = random.random()
        colors[ctr,0] = r
        colors[ctr,1] = b
        colors[ctr,2] = g
        ctr += 1
    #plot points
    ctr = 0
    for row in data:
        x = row[0]
        y = row[1]
        ptr = int(cluster[ctr])
        r = colors[ptr, 0]
        b = colors[ptr, 1]
        g = colors[ptr, 2]
        color = (r,b,g)
        plt.scatter(x, y, c=np.array([color]))
        ctr += 1
    #plot means and display
    c_x, c_y = k_data.T
    plt.scatter(c_x, c_y, c='k')
    plt.show()

dset = dataload(data_set)
run_k_means(dset)