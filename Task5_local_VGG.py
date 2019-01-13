from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
#import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

#from eval import compute_map
#import models

#from tensorflow.core.framework.summary_pb2 import Summary
from tempfile import TemporaryFile
import pdb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

features_pool5 = np.load('Np_pool5_VGG_task5.npy')
features_fc7 = np.load('Np_fc7_VGG_task5.npy')
#data_dir= '/home/bj/Desktop/CMU_homework/Visual_Learning/HW1/HW1_trail/VOCdevkit'
data_dir= 'VOCdevkit'

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


def load_pascal(data_dir="VOCdevkit", split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
    # Wrote this function

    location_folder = data_dir + "/VOC2007/ImageSets/Main/"	
    image_folder = data_dir + "/VOC2007/JPEGImages/"

###################################################################
    images_list = []
    #path = location_folder + '/' + 'aeroplane' + '_'+split+'.txt'
    path = location_folder +  'aeroplane' + '_'+split+'.txt'
    #pdb.set_trace()

    file = open(path,'r')
    for ctr,line in enumerate(file):
        image_file_name = line.split(' ')[0]
        images_list.append(image_file_name) 

    N = len(images_list)
    #images = np.zeros((N,256,256,3))
    labels = np.zeros((N,20))	
    #weights = np.zeros((N,20))



#################################################################	
#    for ctr,image_file in enumerate(images_list):	
#        im = Image.open(image_folder + image_file + '.jpg')
#        im = im.resize((256,256))
#        im_np = np.asarray(im)
#	#name = int(image_file.split('.')[0])
#        images[ctr,:,:,:] = im_np

    for class_number,class_name in enumerate(CLASS_NAMES):
        path = location_folder + class_name + '_' + split + '.txt'
        file = open(path,'r')
        for ctr,line in enumerate(file):
            image_file_name = line.split(' ')[0]
	    #digits = int(line.split(' ')[1].split('\n')[0])
            digits = line.split(' ')[-1]
            if digits == '-1\n':
                labels[ctr,class_number] = 0
                #weights[ctr,class_number] = 1
            if digits == '0\n':
                labels[ctr,class_number] = 1
                #weights[ctr,class_number] = 0
            if digits == '1\n':
                labels[ctr,class_number] = 1
                #weights[ctr,class_number] = 1
        file.close()

    #images = np.asarray(images, dtype=np.float32)	
    #return images,labels,weights

    return images_list,labels
#################################################################	



def load_image(data_dir,image_file):
    image_folder = data_dir + "/VOC2007/JPEGImages/"
    #for ctr,image_file in enumerate(images_list):	
    im = Image.open(image_folder + image_file + '.jpg')
    im = im.resize((256,256))
    im_np = np.asarray(im)
    #name = int(image_file.split('.')[0])
    #images[ctr,:,:,:] = im_np
    #plt.imshow(im_np)
    #plt.imshow(im)
    #plt.show()
    return im
    




images_list,labels = load_pascal(data_dir, split='test')

#im = load_image(data_dir,'000002')
#plt.imshow(im)
#plt.show()

#pdb.set_trace()


#aeroplane - 000067  
#chair - 000008  
#dog - 000205  
#horse - 000166  
#sofa - 000108 
#cow - 000013  
#bird - 000199  
#diningtable - 000084  
#pottedplant - 000070  
#cat - 000011  
test_samples = ['000067','000008','000205','000166','000108','000013', '000199','000084','000070','000011'];	    	
best_match_pool5 = [];
best_match_fc7 = [];


print('Pool5 now')

for sample in test_samples:

    min_dist = 1000000
    min_dist_lablel = 0

    loc = images_list.index(sample)
    test_feature = features_pool5[loc]
    for feat_index,feature in enumerate(features_pool5):
        dist = np.sum((test_feature - feature)**2)
        if(dist < min_dist and dist !=0):
            min_dist = dist
            min_dist_label = feat_index
    best_match_pool5.append(images_list[min_dist_label]) 


print('FC7 now')

for sample in test_samples:

    min_dist = 1000000
    min_dist_lablel = 0

    loc = images_list.index(sample)
    test_feature = features_fc7[loc]

    for feat_index,feature in enumerate(features_fc7):
        dist = np.sum((test_feature - feature)**2)
        if(dist < min_dist and dist !=0):
            min_dist = dist
            min_dist_label = feat_index
    best_match_fc7.append(images_list[min_dist_label]) 

#pdb.set_trace()

########################################################################################

iii=1
for i in range(5):

    print(i)	
    plt.subplot(5,3,iii)
    plt.imshow(load_image(data_dir,test_samples[i]))
    plt.xticks([])
    plt.yticks([])

    iii+=1
    plt.subplot(5,3,iii)
    plt.imshow(load_image(data_dir,best_match_fc7[i]))
    plt.xticks([])
    plt.yticks([])

    iii+=1 
    plt.subplot(5,3,iii)
    plt.imshow(load_image(data_dir,best_match_pool5[i]))
    plt.xticks([])
    plt.yticks([])
    iii+=1

plt.subplots_adjust(wspace=-0.8, hspace=0)
plt.show()

iii=1
for i in range(5,10):

    print(i)	
    plt.subplot(5,3,iii)
    plt.imshow(load_image(data_dir,test_samples[i]))
    plt.xticks([])
    plt.yticks([])

    iii+=1
    plt.subplot(5,3,iii)
    plt.imshow(load_image(data_dir,best_match_fc7[i]))
    plt.xticks([])
    plt.yticks([])

    iii+=1 
    plt.subplot(5,3,iii)
    plt.imshow(load_image(data_dir,best_match_pool5[i]))
    plt.xticks([])
    plt.yticks([])
    iii+=1

plt.subplots_adjust(wspace=-0.8, hspace=0)
plt.show()



########################################################################################
                   

print('\n\nTSINE NOW\n\n')

samples = np.random.permutation(len(labels))[0:1000]
features_fc7 = features_fc7[samples]
labels = labels[samples]

fc7_embedded = TSNE(n_components=2).fit_transform(features_fc7)

#colors = [[0,0,0],[0,0,0.1],[0,0,0.2],[0,0,0.3],[0,0,0.4],[0,0,0.5],[0,0,0.6],[0,0,0.7],[0,0,0.8],[0,0,0.9],[0,0,1],[0,0.1,0],[0,0.2,0],[0,0.3,0],[0,0.4,0],[0,0.5,0],[0,0.6,0],[0,0.7,0],[0,0.8,0],[0,0.9,0],[0,1,0]]

colors = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[0.3,0.4,0.5],[0,0.1,0.8],[0.4,1,0.2],[0,0.9,0.9],[0.1,0.2,0.3],[0.9,0.8,0.5],[0.5,0.6,0.05],[0.35,0.78,0.1],[0.67,0.78,0.5],[0.08,0.67,1],[0.34,0.98,1],[0.32,0.08,0.2],[1,0.2,0.05]]


colors = np.array(sns.color_palette("hls", 20))
sns.palplot(sns.color_palette("hls", 20))
plt.show()

#x = fc7_embedded[:,0]
#y = fc7_embedded[:,1]

for i in range(len(fc7_embedded)):
    	
    pp = np.argmax(labels[i])
    plt.scatter(fc7_embedded[i,0], fc7_embedded[i,1], c=colors[pp], alpha=0.5)	


plt.show()


#colors = [[0,0,1],[0,1,0],[1,0,0],[0,0,0]]


