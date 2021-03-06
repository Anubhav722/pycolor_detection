from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import numpy.ma as ma
import matplotlib.colors as colors
from colormap import rgb2hex
import numpy as np
import time
import sys
from math import sqrt
import random
from collections import defaultdict

'__author__' == "sharmaparth17@gmail.com"

def func(t):
    if (t > 0.008856):
        return np.power(t, 1/3.0)
    else:
        return 7.787 * t + 16 / 116.0

def rgbtolab(requested_color):
    
    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    # RGB values lie between 0 to 1.0
    
    cie = np.dot(matrix, requested_color);
    cie[0] = cie[0] /0.950456;
    cie[2] = cie[2] /1.088754; 

    # Calculate the L
    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];

    # Calculate the a 
    a = 500*(func(cie[0]) - func(cie[1]));

    # Calculate the b
    b = 200*(func(cie[1]) - func(cie[2]));

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
    Lab = [b , a, L]; 

    # OpenCV Format
    L = L * 255 / 100;
    a = a + 128;
    b = b + 128;
    Lab_OpenCV = [b, a, L]; 
    
    return Lab_OpenCV
def match_colour(ccolor):
    #converting hex to rgb   
    requested_color = colors.hex2color(ccolor)
    lab= rgbtolab(requested_color)

    return lab

def data(map_csv_path):
    dfs = pd.read_csv(map_csv_path)
    dfs['hex_to_rgb'] = dfs['HEX Code'].apply(match_colour)

    return dfs

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    # return the bar chart
    return bar

def closest_colour(requested_colour,dfs):
    min_colours = {}

    requested_color = colors.hex2color(requested_colour)
    
    requested_color = rgbtolab(requested_color)
    # K-Nearest Neighbours implementation using Euclidean distance
    for key,color_shade,code,color_base in zip(dfs['hex_to_rgb'],dfs['Color Name'],dfs['HEX Code'],dfs['Base Color']):
        r_c ,g_c,b_c = map(float,key)
        rd = (r_c - float(requested_color[0])) ** 2
        gd = (g_c - float(requested_color[1])) ** 2
        bd = (b_c - float(requested_color[2])) ** 2
        min_colours[sqrt(rd + gd + bd)] = [color_shade,code,color_base]
    return min_colours[min(min_colours.keys())]




def detect_color(img_byte_array,map_path):
    color_csv=[]
    hexcod_csv=[]
    actualhexcod_csv=[]
    closest_nameshade=dict()
    closest_namebase = dict()
    percentcsv=[]
    totalcsv_shade=0.0
    totalcsv_base=0.0
    cluster_errors = []
    domcol=[]
    base = []
    dom_array = []
    dfs = data(map_path)
    final_colorcsv = defaultdict(list)
    final_colorbase = defaultdict(list)
    img = cv2.imdecode(np.squeeze(np.asarray(img_byte_array[1])),-1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], 3))
    img_final = ma.masked_where(img_reshaped == [0, 0, 0], img_reshaped)
    
    for clusters in xrange(3, 17):
        
        # Cluster colours
        clt = MiniBatchKMeans(n_clusters = clusters,random_state=2)
        clt.fit(img_final)
     
        # Validate clustering result
        cluster_errors.append( clt.inertia_ )
    
    bestClusters = cluster_errors.index(min(cluster_errors))+3
        
    # Find the best one
    # bestClusters = cluster_errors.index(min(cluster_errors)+1)

    for index in xrange(2):
        clt = MiniBatchKMeans(n_clusters=bestClusters,random_state=2,max_iter=200)
        clt.fit(img_final)
        hist = centroid_histogram(clt)
        for (percent, color) in zip(hist, clt.cluster_centers_):
            
            requested_colour = color
            if  int(requested_colour[0]) > 0 and int(requested_colour[1])> 0 and int(requested_colour[2]) >0:
                hexcod = rgb2hex(int(requested_colour[0]),int(requested_colour[1]),int(requested_colour[2]))
                actualhexcod_csv.append(hexcod)
                output = closest_colour(hexcod,dfs)
                color_shade,hexcoda,color_base = output[0],output[1],output[2]
                color_csv.append(requested_colour)
                # print color_shade
                if color_shade not in closest_nameshade.keys():
                    closest_nameshade[color_shade] = 1
                    closest_namebase[color_shade] = [color_base]
                else:
                    closest_nameshade[color_shade] += 1
                
                hexcod_csv.append(hexcoda)
                percentcsv.append(round(percent,2)*100)
                    # totalcsv_shade+=float(round(percent,2)*100) 

    out = sum(closest_nameshade.values())
    for key in closest_nameshade.keys():
        final_colorcsv[(float(closest_nameshade[key])/out)*100] += [key]
    final_shade = list()
    shade = final_colorcsv[max(final_colorcsv.keys())]
    
    for s in shade:
        if closest_namebase[s][0] not in base:
            base += closest_namebase[s]
            final_shade.append(s)
    
    

    
    # del final_colorcsv[max(final_colorcsv.keys())]
    # shade = final_colorcsv[max(final_colorcsv.keys())]
    
    # for s in shade:
    #   # print '2',closest_namebase[s]
    #   if closest_namebase[s][0] not in base:
    #       base += closest_namebase[s]
    #       final_shade.append(s)
    
    # print base,final_shade
    for S,B in zip(final_shade,base) :
        dom_array.append(
            {
                'prediction': 
                {
                    'Shade': S,
                    'Base': B
                },
            }
        )

    return dom_array
    