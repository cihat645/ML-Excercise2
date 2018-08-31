import numpy as np, pandas as pd, random
import matplotlib.pyplot as plt, sys

def Initialize_Centroids(Data, K):
    random_indices = random.sample(range(0,Data.shape[0]), K)   # generate K random indices
    return Data[random_indices,:]

def Centroid_Assignment(Centroids, Data):
    """Assigns each data example to the nearest centroid on each iteration.
    Input:  Centroids - a numpy matrix with dimensionality k x n where k = # of clusters and n = number of features / dimensionality of each training example
            Data - numpy matrix with dimensionality m x n (m = # training examples)
    Output: an m x 1 array of integers indicating centroid assignment for each training example"""
    indices = np.zeros((Data.shape[0],1))
    r = 0
    for row in Data:  # iterate a numpy matrix by row by default
        min_dist = sys.float_info.max
        for k in range(Centroids.shape[0]):
            new_dist = np.linalg.norm(row - Centroids[k,:])
            if new_dist < min_dist:
                min_dist = new_dist
                k_val = k
        indices[r] = k_val
        r += 1
    return indices

def Update_Centroids(indices, Data, K):
    """Moves each centroid to the mean of all the data points assigned to it"""
    new_centroids = np.zeros((K, Data.shape[1]))
    for k in range(K):
        new_centroids[k,:] = Data[[i for i in range(len(indices)) if indices[i] == k], :].mean(axis=0) # select all rows of Data where indices[i] == k, then computing row vector whose elements are the mean of each col
        # note: .mean(axis = 0) specifies we are taking the col mean, whereas axis = 1 corresponds to the row mean
    return new_centroids

def KMeans(Data, K, iter = 10):
    # centroids = Initialize_Centroids(Data,K)                # dimen: K x n where n = Data.shape[1] (i.e. number of columns)
    centroids = np.array([(3, 3), (6, 2), (8, 5)]) # for sanity checking
    centroid_tensor = np.zeros((iter, K, Data.shape[1]))
    centroid_tensor[0,:,:] = centroids
    for i in range(1,iter):
        print("K-Means Iteration ", str(i) + "...")
        cluster_assignments = Centroid_Assignment(centroids, Data)      # cluster assignments is a m x 1 array (maps each training example to a cluster)
        centroids = Update_Centroids(cluster_assignments, Data, K)
        centroid_tensor[i,:,:] = centroids                          # store the current centroid matrix in the tensor

    all_clusters = list()
    for k in range(K):
        all_clusters.append(np.array(Data[[x for x in range(len(cluster_assignments)) if cluster_assignments[x] == k],:]))

    Show_Centroid_Path(centroid_tensor, all_clusters)
    return all_clusters             # return clusters of data for analysis

def Show_Centroid_Path(centroid_tensor, finalized_clusters):
    """ This function takes in a matrix of the centroid history and displays the origin with a pentagon and the ending point with a circle.
    This implementation is for visualizing 2D data.
    input: centroids, a tensor which is (iter x K x n). K = num of clusters, n = num of features (columns) of Data, iter = num iterations of K means
    output: None

    Note on tensors: T[i,r,c] indexes the rth row and cth column of the ith matrix of the tensor T. T.shape = (m,r,c). This indicates T is composed of m matrices,
        each of dimensions (r x c).
    """

    if centroid_tensor.shape[1] > 8:            # if K > 8
        print("Centroid tensor has ", centroid_tensor.shape[1], "clusters. Exceeded maximum K = 8")
        return

    fig = plt.figure()
    colors = 'r b g c m y k w'.split()  # 8 colors for 8 different clusters

    # centroid_tensor[i,:,:] is the matrix of centroids with dimensions k x n for iteration i

    m = 0           # m indexes each each artificially generated cluster
    for CHM in centroid_tensor[0,:,:]:  # Plotting origin for each CHM
        plt.plot(CHM[0], CHM[1], marker='o', color=colors[m], label="Centroid" + str(m + 1))
        m += 1

    for CHM in centroid_tensor[centroid_tensor.shape[0] - 1,:,:]:  # Plotting end point for each CHM
        plt.plot(CHM[0], CHM[1], marker='X', color= 'k', markersize = 8)


    m = 0
    for k in range(centroid_tensor.shape[1]):                                       # plotting the rest of the paths and the data points corresponding to each cluster
        plt.plot(centroid_tensor[:,k,0], centroid_tensor[:,k,1], color = colors[m])
        cluster_k = finalized_clusters[k]
        plt.scatter(cluster_k[:,0], cluster_k[:,1], marker = '.', color = colors[m])
        m += 1



    plt.legend(loc=4)
    plt.show()

def create_clusters():
    x_cluster1 = np.random.normal(5, 1, (10, 1))
    y_cluster1 = np.random.normal(2, 0.5, (10, 1))
    x_cluster2 = np.random.normal(0, 1, (25, 1))
    y_cluster2 = np.random.normal(10, 1.5, (25, 1))
    x_cluster3 = np.random.normal(-3, 0.25, (16, 1))
    y_cluster3 = np.random.normal(-3, 1.5, (16, 1))
    cluster1 = np.column_stack((x_cluster1, y_cluster1))
    cluster2 = np.column_stack((x_cluster2, y_cluster2))
    cluster3 = np.column_stack((x_cluster3, y_cluster3))
    return (np.row_stack((cluster1,cluster2,cluster3)), cluster1, cluster2, cluster3)

def plot_clusters(cluster1, cluster2, cluster3):  # need to generalize this function for N clusters
    fig = plt.figure()
    plt.scatter(list(cluster1[:, 0]), list(cluster1[:, 1]), color='red', label="Cluster 1")
    plt.scatter(list(cluster2[:, 0]), list(cluster2[:, 1]), color='blue', label="Cluster 2")
    plt.scatter(list(cluster3[:, 0]), list(cluster3[:, 1]), color='green', label="Cluster 3")
    plt.title("My random clusters")
    plt.legend(loc=1)
    plt.show()

def TensorDemo():
    # TENSOR BEHAVIOR / INDEXING
    # - Creating 3 matrices which correspond to sample centroid positions over all K-means iterations. In this example, we have 2D data over 4 iterations for 3 cluster centroids
    cent1 = np.array([(1, 2), (6, 4), (3, 3), (5, 3)])
    cent2 = np.array([(0, 0), (-3, 1), (-1, -2), (-2, -1)])
    cent3 = np.array([(5, 5), (2, 2), (1, 1), (1, 2)])
    # we have 3 CHMs (Centroid history matrices), each with dimensions (iter x n) where iter = num iterations and n = number of features of data.
    # Now we stack them to form a rank 3 tensor
    CentroidTensor = np.array([cent1, cent2, cent3])
    print(CentroidTensor.shape)
    print("There are ", CentroidTensor.shape[0], " matrices in this tensor")
    print("There are ", CentroidTensor.shape[1], " rows for each matrix in this tensor")
    print("There are ", CentroidTensor.shape[2], " columns for each matrix in this tensor")

    # Illustrating default iteration behavior: iterates over tensor by matrix first
    for CHMI in range(CentroidTensor.shape[0]): # CHMI = centroid history matrix index, explicitly iterating over each matrix
        print(CentroidTensor[CHMI,:,:])

    for CHM in CentroidTensor:      # implicitly iterating over each matrix
        print(CHM)

    print("\n\n\n")

    for CHM in CentroidTensor:      # implicitly iterating over each matrix
        print("A Matrix: ")
        for row in CHM:             # implicitly iterating over each row of matrix CHM. If we were to do another for loop, it would implicitly iterate over each column.
            print(row)

    print("\n\n\n")

    fig = plt.figure()
    colors = 'r b g c m y k w'.split()  # 8 colors for 8 different clusters

    m = 0
    for CHM in CentroidTensor:  # Plotting origin for each CHM
        plt.plot(CHM[0, 0], CHM[0, 1], marker='o', color=colors[m], label="Centroid" + str(m + 1))
        m += 1

    m = 0
    for CHM in CentroidTensor:  # plotting the connections between all centroid locations
        plt.plot(CHM[:, 0], CHM[:, 1], color=colors[m])
        m += 1
        if m == 8:
            print("Too many clusters")
            break
    m = 0
    for CHM in CentroidTensor:  # Plotting final location for each centroid
        plt.plot(CHM[CHM.shape[0] - 1, 0], CHM[CHM.shape[0] - 1, 1], marker='p', color=colors[m])
        m += 1

    plt.legend(loc=4)
    plt.show()


# MAIN PROGRAM:

# sanity check
import scipy.io
Data = scipy.io.loadmat('/Users/thomasciha/Documents/Summer Research 2018/machine-learning-ex7/ex7/ex7data2.mat')
Data = Data['X']
cluster_results = KMeans(Data,3)
cluster0 = cluster_results[0]
cluster1 = cluster_results[1]
cluster2 = cluster_results[2]


# Trying on randomly generated clusters
Data, c1, c2, c3 = create_clusters()
cluster_results = KMeans(Data,3)


