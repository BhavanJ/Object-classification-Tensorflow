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

import os
import pdb

BATCH_SIZE = 10
NUM_ITERS = 1000

tf.logging.set_verbosity(tf.logging.INFO)

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



def cnn_model_fn(features, labels, mode, num_classes=20):

    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 64 * 64 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    #logits = tf.sigmoid(tf.layers.dense(inputs=dropout, units=20))		###############VERIFY HERE
    logits = tf.layers.dense(inputs=dropout, units=20)		###############VERIFY HERE

    predictions = {							
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")		###########VERIFY HERE
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, weights = features["w"], logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
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
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="./tmp/pascal_01")

    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    pascal_classifier.train(
        input_fn=train_input_fn,
        steps=NUM_ITERS,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)



    writer = tf.summary.FileWriter("./tmp/pascal_01")

    for ii in range(10):
        pascal_classifier.train(input_fn=train_input_fn,steps=NUM_ITERS/10,hooks=[logging_hook])				
	
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print(ii)
        print('Obtained {} mAP'.format(np.mean(AP)))
	
        summary = tf.Summary(value=[tf.Summary.Value(tag='mAP', simple_value=np.mean(AP))])
        writer.add_summary(summary, (ii+1)*NUM_ITERS/10)


    writer.close()


    print('\n\nTesting now...done 1000 itterations\n\n')	

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
