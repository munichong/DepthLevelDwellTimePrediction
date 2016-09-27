'''
Created on Jan 23, 2016

@author: Wang
'''
from pymongo import MongoClient 
from collections import defaultdict
from numpy import array, mean
import numpy as np
# import pylab as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist


client = MongoClient()
featVec_DB = client['Forbes_Dec2015']['FreqUserLogPV']



def dwelltime_trend_visualization():
    matrix = []
    n = 0  
    logs = []  
    for pv_doc in featVec_DB.find():
    #     ''' skip pageviews which only has 0 or 1 log '''
    #     if len(pv_doc['loglist']) <= 1:
    #         continue
        print(n)
        n += 1
        
        if n > 6000:
            break
        
        pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
        skip_pageview = False
        for index, log_dict in enumerate(pv_doc['loglist']):
            additionalinfo = log_dict['additionalinfo']
            
            if log_dict["eventname"] == "Post reading":
                if 'Percentage reading from' not in additionalinfo:
                    skip_pageview = True
                    break
                screen_top = additionalinfo['Percentage reading from']
                screen_bottom = additionalinfo['Percentage of reading']
                dwell_time = additionalinfo['Time on article']
                if dwell_time > 300:
                    skip_pageview = True
                    break
                
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if index < len(pv_doc['loglist']) - 1:
                    pv_summary.append([screen_top, screen_bottom, dwell_time])
                    
            elif log_dict["eventname"] == "Text selection":
                continue    
            elif log_dict["eventname"] == "Probably left":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Load new article":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Page leave":
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                else:
                    skip_pageview = True
                break
            else:
                continue
        
        logs.append(pv_summary)
        dwell_collector = defaultdict(list)
        if not skip_pageview:
            ''' Content area overlap is allowed '''
            for screen_top, screen_bottom, dwell_time in pv_summary:
                for depth in range(screen_top, screen_bottom + 1):
                    dwell_collector[depth].append(dwell_time)    
            
            vector = [0] * 101
            for depth in range(101):
                vector[depth] = mean(dwell_collector[depth]) if depth in dwell_collector else 0
            matrix.append(vector[1:])
    
    
    raw_matrix = array(matrix).astype(float)
    
    
    ''' Normalize '''
    matrix = normalize(raw_matrix, norm='l1', axis=1)
    print(matrix)
    print(matrix.shape)
#     print(np.sum(matrix, axis=1))
    print()
    
    ''' elbow '''
    K = range(1,10)
    KM = [kmeans(matrix, k) for k in K]
    centroids = [cent for (cent,var) in KM]
#     print(len(centroids))
#     print(centroids.shape)
    D_k = [cdist(matrix, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/matrix.shape[0] for d in dist]
    print(avgWithinSS)
    print()
#     kIdx = 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
#     ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
#         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
    
    
#     ''' eigenvalues '''
#     mean_vector = []
#     m = matrix.T
#     print(m.shape)
#     for c in range(m.shape[0]):
#         mean_vector.append([np.mean(m[c,:])])
#     mean_vector = np.array(mean_vector)
#     print('Mean Vector:\n', mean_vector)
#     print()
#     
#     scatter_matrix = np.zeros((100,100))
#     for i in range(m.shape[1]):
#         scatter_matrix += (m[:,i].reshape(100,1) - mean_vector).dot((m[:,i].reshape(100,1) - mean_vector).T)
#     print('Scatter Matrix:\n', scatter_matrix)
#     
# #     cov_vector = []
# #     for c in range(m.shape[0]):
# #         cov_vector.append(m[c,:])
# #     cov_mat = np.cov(np.array(cov_vector))
# #     print('Covariance Matrix:\n', cov_mat)
# #     print(cov_mat.shape)
# 
#     eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
#     print(eig_val_sc)
#     eig_val_sc = sorted(eig_val_sc)
#     print((eig_val_sc[-1] + eig_val_sc[-2]) / np.sum(eig_val_sc)) # 0.776156859312
    
    
    ''' PCA '''
    pca = PCA(n_components=2)
    matrix_2d = pca.fit_transform(matrix)
    print(matrix)
    print()
    
    ''' MDS '''
#     mds = MDS(n_components=2)
#     matrix_2d = mds.fit_transform(matrix)
#     print(matrix)
#     print()
    
    ''' 2D '''
#     plt.scatter(matrix.T[0], matrix.T[1], color='blue', marker='.')
#     plt.show()
    
#     plt.hist(new_matrix.T[0], 100)
#     plt.show()
    
#     plt.hist(new_matrix.T[1], 100)
#     plt.show()
    
    
    '''3D'''
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(new_matrix.T[0], new_matrix.T[1], new_matrix.T[2], marker='.')
#     plt.show()
    
#     matrix_2d = matrix
    ''' plot clustering '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    handles = []
#     print(cIdx[3])
    colors = ['blue', 'red', 'green', 'magenta', 'purple']
    handles.append(ax.scatter(matrix_2d.T[0][cIdx[4] == 0], matrix_2d.T[1][cIdx[4] == 0],
                              color='blue', marker='.'))
    handles.append(ax.scatter(matrix_2d.T[0][cIdx[4] == 1], matrix_2d.T[1][cIdx[4] == 1],
                              color='red', marker='.'))
    handles.append(ax.scatter(matrix_2d.T[0][cIdx[4] == 2], matrix_2d.T[1][cIdx[4] == 2],
                              color='green', marker='.'))
    handles.append(ax.scatter(matrix_2d.T[0][cIdx[4] == 3], matrix_2d.T[1][cIdx[4] == 3],
                              color='magenta', marker='.'))
    handles.append(ax.scatter(matrix_2d.T[0][cIdx[4] == 4], matrix_2d.T[1][cIdx[4] == 4],
                              color='purple', marker='.'))
    plt.grid(True)
    plt.show()
            
        
    ''' individual check '''
    example_blue = matrix[cIdx[4] == 0]
    example_red = matrix[cIdx[4] == 1]
    example_green = matrix[cIdx[4] == 2]
    example_magenta = matrix[cIdx[4] == 3]
    example_purple = matrix[cIdx[4] == 4]
    print('\nBLUE:')
    print(example_blue[:10])
    print('\nRED:')
    print(example_red[:10])
    print('\nGREEN:')
    print(example_green[:10])
    print('\nMAGENTA:')
    print(example_magenta[:10])
    print('\nPURPLE:')
    print(example_purple[:10])
    
    
    ''' plot centroids '''
    centroids_k = centroids[4]
    print(len(centroids_k), 'centroids')
    for cent in centroids_k:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0,100])
        ax.set_ylim([0,0.08])
        ax.plot(range(1, 101), cent, 'b*-')
        plt.grid(True)
        plt.show()


    ''' Check Last Dwell Time VS. Current Dwell Time of A Cluster'''
    for cluster_index in range(4+1):
        last_dwells = []
        current_dwells = []
        print(colors[cluster_index])
        
        for instance_index in range(len(cIdx[4])):
            in_cluster = cIdx[4][instance_index] == cluster_index
            if not in_cluster:
                continue
            for action_index in range(len(logs[instance_index])):
                # if this action is the last one in this pageview
                if action_index+1 == len(logs[instance_index]):
                    break
                if logs[instance_index][action_index+1][1] > 50:
                    break
                last_dwells.append(logs[instance_index][action_index][2])
                current_dwells.append(logs[instance_index][action_index+1][2])
            print(logs[instance_index])
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0,50])
        ax.set_ylim([0,50])
        ax.scatter(last_dwells, current_dwells, color='blue', marker='.')
        plt.grid(True)
        plt.show()
        
        # print points
#         for i in range(len(last_dwells)):
#             print("%f,%f" % (last_dwells[i], current_dwells[i]))
#         print()

    

if __name__ == "__main__":
    dwelltime_trend_visualization()
    
    
    