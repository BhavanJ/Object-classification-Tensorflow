from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
#import models

from tensorflow.core.framework.summary_pb2 import Summary



tf.logging.set_verbosity(tf.logging.INFO)


BATCH_SIZE = 10
NUM_ITERS = 4000
alpha_val = 0.1
 
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

mapping = { 'vgg_16/conv1/conv1_1/biases': 'vgg16/conv1_1/bias',
            'vgg_16/conv1/conv1_1/weights': 'vgg16/conv1_1/kernel',
            'vgg_16/conv1/conv1_2/biases': 'vgg16/conv1_2/bias',
            'vgg_16/conv1/conv1_2/weights': 'vgg16/conv1_2/kernel',
            'vgg_16/conv2/conv2_1/biases': 'vgg16/conv2_1/bias',
            'vgg_16/conv2/conv2_1/weights': 'vgg16/conv2_1/kernel',
            'vgg_16/conv2/conv2_2/biases': 'vgg16/conv2_2/bias',
            'vgg_16/conv2/conv2_2/weights': 'vgg16/conv2_2/kernel',
            'vgg_16/conv3/conv3_1/biases': 'vgg16/conv3_1/bias',
            'vgg_16/conv3/conv3_1/weights': 'vgg16/conv3_1/kernel',
            'vgg_16/conv3/conv3_2/biases': 'vgg16/conv3_2/bias',
            'vgg_16/conv3/conv3_2/weights': 'vgg16/conv3_2/kernel',
            'vgg_16/conv3/conv3_3/biases': 'vgg16/conv3_3/bias',
            'vgg_16/conv3/conv3_3/weights': 'vgg16/conv3_3/kernel',
            'vgg_16/conv4/conv4_1/biases': 'vgg16/conv4_1/bias',
            'vgg_16/conv4/conv4_1/weights': 'vgg16/conv4_1/kernel',
            'vgg_16/conv4/conv4_2/biases': 'vgg16/conv4_2/bias',
            'vgg_16/conv4/conv4_2/weights': 'vgg16/conv4_2/kernel',
            'vgg_16/conv4/conv4_3/biases': 'vgg16/conv4_3/bias',
            'vgg_16/conv4/conv4_3/weights': 'vgg16/conv4_3/kernel',
            'vgg_16/conv5/conv5_1/biases': 'vgg16/conv5_1/bias',
            'vgg_16/conv5/conv5_1/weights': 'vgg16/conv5_1/kernel',
            'vgg_16/conv5/conv5_2/biases': 'vgg16/conv5_2/bias',
            'vgg_16/conv5/conv5_2/weights': 'vgg16/conv5_2/kernel',
            'vgg_16/conv5/conv5_3/biases': 'vgg16/conv5_3/bias',
            'vgg_16/conv5/conv5_3/weights': 'vgg16/conv5_3/kernel',
            'vgg_16/fc6/biases': 'vgg16/fc6/bias',
            'vgg_16/fc6/weights': 'vgg16/fc6/kernel',
            'vgg_16/fc7/biases': 'vgg16/fc7/bias',
            'vgg_16/fc7/weights': 'vgg16/fc7/kernel'}


#final_layer/kernel
#final_layer/bias




class RestoreHook(tf.train.SessionRunHook):
    def begin(self):
        trainable_variables = tf.trainable_variables()
        for v in trainable_variables:
            vname = v.name.replace(":0", "")
            print(vname)
        tf.train.init_from_checkpoint("vgg_16.ckpt", mapping)



#weights_dictsdfsdf = {vgg_16ckpt names to model names....from get_variable_name()}

#class LoadInitializersHooksdfsf(tf.train.SessionRunHook):
#    def begin(self):
#        tf.contrib.framework.init_from_checkpoint('vgg_16.ckpt',weights_dictsdfsdf)

#https://www.tensorflow.org/api_docs/python/tf/train/init_from_checkpoint


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
    images = np.zeros((N,256,256,3))
    labels = np.zeros((N,20))	
    weights = np.zeros((N,20))



#################################################################	
    for ctr,image_file in enumerate(images_list):	
        im = Image.open(image_folder + image_file + '.jpg')
        im = im.resize((256,256))
        im_np = np.asarray(im)
	#name = int(image_file.split('.')[0])
        images[ctr,:,:,:] = im_np

    for class_number,class_name in enumerate(CLASS_NAMES):
        path = location_folder + class_name + '_' + split + '.txt'
        file = open(path,'r')
        for ctr,line in enumerate(file):
            image_file_name = line.split(' ')[0]
	    #digits = int(line.split(' ')[1].split('\n')[0])
            digits = line.split(' ')[-1]
            if digits == '-1\n':
                labels[ctr,class_number] = 0
                weights[ctr,class_number] = 1
            if digits == '0\n':
                labels[ctr,class_number] = 1
                weights[ctr,class_number] = 0
            if digits == '1\n':
                labels[ctr,class_number] = 1
                weights[ctr,class_number] = 1
        file.close()

    images = np.asarray(images, dtype=np.float32)	

    return images,labels,weights
#################################################################

#https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way
#https://stackoverflow.com/questions/38920240/tensorflow-image-operations-for-batches
#https://www.tensorflow.org/api_docs/python/tf/map_fn
#https://github.com/tensorflow/transform/blob/master/getting_started.md
#https://www.tensorflow.org/versions/r0.12/api_docs/python/image/cropping
#https://stackoverflow.com/questions/43997719/how-to-crop-tensor-in-the-center-in-tensorflow


def cnn_model_fn(features, labels, mode, num_classes=20):

    #print('\n\nSize of lables\n\n')	
    #print(labels)
    #print(features)
    #print('\n\n')

    if mode == tf.estimator.ModeKeys.TRAIN:										########VERIFY


        lamb = np.random.beta(alpha_val,alpha_val)  

        features_u = tf.unstack(features["x"])
        labels_u = tf.unstack(labels)

        for jj in range(BATCH_SIZE):  
            if (jj==BATCH_SIZE-1): 	
                features_u[jj] = lamb*features_u[jj] + (1.0-lamb)*features_u[0]
                labels_u[jj] = lamb*labels_u[jj] + (1.0-lamb)*labels_u[0]
            else:	
                features_u[jj] = lamb*features_u[jj] + (1.0-lamb)*features_u[jj+1]
                labels_u[jj] = lamb*labels_u[jj] + (1.0-lamb)*labels_u[jj+1]
            	
        features["x"] = tf.stack(features_u)
        labels = tf.stack(labels_u)

        transformed_features = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), features["x"])
        transformed_features = tf.map_fn(lambda img: tf.random_crop(img,[224,224,3]),transformed_features)

    if mode == tf.estimator.ModeKeys.PREDICT:									########VERIFY
        transformed_features=tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img,224,224), features["x"])
        #transformed_features=tf.map_fn(lambda img: tf.image.central_crop(img,(224/256)), features["x"])

    if mode == tf.estimator.ModeKeys.EVAL:										########VERIFY
        transformed_features=tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img,224,224), features["x"])
        #transformed_features=tf.map_fn(lambda img: tf.image.central_crop(img,(224/256)), features["x"])

    #print('\n\n')
    #print(transformed_features)
    #print('\n\n')


#features["x"]  = tf.image.central_crop(features["x"], central_fraction)


    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(transformed_features, [-1, 224, 224, 3])   				########VERIFY

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv1_1',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	#kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 			######VERIFY
	bias_initializer=tf.zeros_initializer())									######VERIFY


    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv1_2',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,)

    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv2_1',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv2_2',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())


    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv3_1',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv3_2',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv3_3',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv4_1',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv4_2',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv4_3',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv5_1',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv5_2',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
#	kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        activation=tf.nn.relu, name = 'vgg16/conv5_3',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	#kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)


    #print('\n\n')
    #print(pool5)
    #print('\n\n')


    # Dense Layer
####pool5_flat = tf.reshape(pool5,[-1,7*7*512]) ###############VERIFY HERE
#dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu, bias_initializer=tf.zeros_initializer())
#dense1 to cpnv2d (kermel = 7x7, stride..don't define, "valid" )
#dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu, bias_initializer=tf.zeros_initializer())
#dense2 to cpnv2d (kerenle = 1x1, stride= , "valid")


    dense1 = tf.layers.conv2d(
        inputs=pool5,
        filters=4096,
        kernel_size=[7, 7],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu, name = 'vgg16/fc6',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	#kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())

    #print('\n\n\n\n')	
    #print(dense1.shape)


    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


    dense2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=4096,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding="valid",
        activation=tf.nn.relu, name = 'vgg16/fc7',
#	kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	#kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32), 
	bias_initializer=tf.zeros_initializer())



    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    #print('\n\n\n\n')	
    #print(dropout2.shape)
	

    dropout2_flat = tf.reshape(dropout2,[-1,4096])	

    #print('\n\n')
    #print(pool5_flat)
    #print('\n\n')


    #print('\n\n')
    #print(dropout2)
    #print('\n\n')


###################################################################################################################
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2_flat, units=20, bias_initializer=tf.zeros_initializer(), name = 'final_layer')		###############VERIFY HERE

    predictions = {							
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")		###########VERIFY HERE
    }

    #print('\nMOdel parameters\n\n ')	
    #print(tf.trainable_variables())



    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

	#global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate,tf.train.get_global_step(),1000, 0.5)		#####VERIFY HERE

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)			#####VERIFY HERE

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        #tf.summary.histogram(name = "Conv1_HOG", values = tf.gradients(loss,conv1)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv2_HOG", values = tf.gradients(loss,conv2)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv3_HOG", values = tf.gradients(loss,conv3)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv4_HOG", values = tf.gradients(loss,conv4)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv5_HOG", values = tf.gradients(loss,conv5)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv6_HOG", values = tf.gradients(loss,conv6)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv7_HOG", values = tf.gradients(loss,conv7)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv8_HOG", values = tf.gradients(loss,conv8)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv9_HOG", values = tf.gradients(loss,conv9)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv10_HOG", values = tf.gradients(loss,conv10)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv11_HOG", values = tf.gradients(loss,conv11)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv12_HOG", values = tf.gradients(loss,conv12)[0]) 					############################ADDED
        #tf.summary.histogram(name = "Conv13_HOG", values = tf.gradients(loss,conv13)[0]) 					############################ADDED

        #tf.summary.scalar(name = "Learning_rate", tensor = learning_rate)						############################ADDED
        #tf.summary.image(name="Images", tensor = transformed_features, max_outputs = 1)					################ADDED	


        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)






def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn, num_classes=train_labels.shape[1]),
        model_dir="./tmp/Task6_VGG",
        config = tf.estimator.RunConfig(model_dir = "./tmp/Task6_VGG", 
        save_summary_steps=200, save_checkpoints_steps = 500)) 

    print(mapping)


    #tensors_to_log = {"loss": "loss"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    
    hook_restore = RestoreHook()

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    writer = tf.summary.FileWriter("./tmp/Task6_VGG")
#########################################################################################################################################################UPDATED####################
    pascal_classifier.train(input_fn=train_input_fn,steps=NUM_ITERS/10,hooks=[hook_restore])       #logging_hook])	########ADDED summary_hook


    for ii in range(9):
        pascal_classifier.train(input_fn=train_input_fn,steps=NUM_ITERS/10)       #logging_hook])	########ADDED summary_hook
	
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
	
        summary = tf.Summary(value=[tf.Summary.Value(tag='mAP', simple_value=np.mean(AP))])
        writer.add_summary(summary, (ii+2)*NUM_ITERS/10)

    writer.close()



    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred = np.stack([p['probabilities'] for p in pred])
    rand_AP = compute_map(
        eval_labels, np.random.random(eval_labels.shape),
        eval_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = compute_map(
        eval_labels, eval_labels, eval_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    AP = compute_map(eval_labels, pred, eval_weights, average=None)
    print('Obtained {} mAP'.format(np.mean(AP)))
    print('per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, _get_el(AP, cid)))

if __name__ == "__main__":
    main()
