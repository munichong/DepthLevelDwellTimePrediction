'''
Created on Jul 18, 2016

@author: munichong
'''
'''
Created on Jan 23, 2016

@author: Wang
'''
from pymongo import MongoClient 

from numpy import array
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
        
        if n > 5000:
            break
        
#         if n == 1507:
#             print()
        
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
                if dwell_time > 180:
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
        
        if not pv_summary:
            continue
        
        logs.append(pv_summary)
        vector = []
        if not skip_pageview:
            vector = [dwell for top, bottom, dwell in pv_summary]
            if len(vector) < 10:
                vector += [0] * (10 - len(vector))
            else:
                vector = vector[:10]
            
            if vector == [0] * 10:
                continue
            matrix.append(vector)
            print(vector)
            
    matrix = array(matrix).astype(float)
    
    
    ''' Normalize '''
    matrix = normalize(matrix, norm='l1', axis=1)
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
    
    
    ''' PCA '''
    pca = PCA(n_components=2)
    matrix_2d = pca.fit_transform(matrix)
    print(matrix)
    print()
    
    
    ''' 2D '''
#     plt.scatter(matrix.T[0], matrix.T[1], color='blue', marker='.')
#     plt.show()
    
#     plt.hist(new_matrix.T[0], 100)
#     plt.show()
    
#     plt.hist(new_matrix.T[1], 100)
#     plt.show()
    
    
    
#     matrix_2d = matrix
    ''' plot clustering '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    handles = []
#     print(cIdx[3])
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
#     example_blue = matrix[cIdx[3] == 0]
#     example_red = matrix[cIdx[3] == 1]
#     example_green = matrix[cIdx[3] == 2]
#     example_magenta = matrix[cIdx[3] == 3]
#     print('\nBLUE:')
#     print(example_blue[:3])
#     print('\nRED:')
#     print(example_red[:3])
#     print('\nGREEN:')
#     print(example_green[:3])
#     print('\nMAGENTA:')
#     print(example_magenta[:3])
    
    
    ''' plot centroids '''
    centroids_k = centroids[4]
    print(len(centroids_k), 'centroids')
    for cent in centroids_k:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, 11), cent, 'b*-')
        plt.grid(True)
        plt.show()

    
    

    ''' Check Last Dwell Time VS. Current Dwell Time of a cluster'''
    for cluster_index in range(4+1):
        last_dwells = []
        current_dwells = []
        instances = matrix[cIdx[cluster_index] == 0]
        for vector in instances:
            for action_index in range(len(vector)):
                if action_index+1 == len(vector):
                    break
                last_dwells.append(vector[action_index])
                current_dwells.append(vector[action_index+1])
                
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0,20])
        ax.set_ylim([0,20])
        ax.scatter(last_dwells, current_dwells, color='blue', marker='.')
        plt.grid(True)
        plt.show()
        
        for i in range(len(last_dwells)):
            print("%f,%f" % (last_dwells[i], current_dwells[i]))
        print()

    

if __name__ == "__main__":
    dwelltime_trend_visualization()
    
    
    