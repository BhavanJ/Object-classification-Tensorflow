# Object-classification-Tensorflow
Visual Learning &amp; Recognition Assignment 1: Object Classification with TensorFlow

Aim: Train multi-label image classification models using the [TensorFlow](www.tensorflow.org) (TF) framework. Classify images from the PASCAL 2007 dataset into the objects present in the image. 


## Task 0: MNIST 10-digit classification in TensorFlow
Run code: python 00_mnist.py

## Task 1: 2-layer network for PASCAL multi-label classification 
#### Q 1.1: Write a data loader for PASCAL 2007.
#### Q 1.2: Modify the MNIST model function to be suitable for multi-label classification.
#### Q 1.3: Same as before, show the training loss and test accuracy (mAP) curves. Train for only 1000 iterations.
Run code: python 01_pascal.py VOCdevkit

## Task 2: Lets go deeper! AlexNet for PASCAL classification
#### Q 2.1: Replace the MNIST model we were using before with this model.
#### Q 2.2: Implement the above solver parameters. This should require small changes to our previous code.
#### Q 2.3: Implement the data augmentation and generate the loss and mAP curves as before.
Run code: python 02_pascal_alexnet.py VOCdevkit

## Task 3: Even deeper! VGG-16 for PASCAL classification
#### Q 3.1: Modify the network architecture from Task 2 to implement the VGG-16 architecture (refer to the original paper). Use the same hyperparameter settings from Task 2, and try to train the model. Add the train/test loss/accuracy curves into the report.
#### Q 3.2: The task in this section is to log the following entities: a) Training loss, b) Learning rate, c) Histograms of gradients, d) Training images and e) Network graph into tensorboard. Add screenshots from your tensorboard into the report.
Run code: python 03_pascal_vgg16.py VOCdevkit

## Task 4: Standing on the shoulder of the giants: finetuning from ImageNet
#### Q 4.1: Load the pre-trained weights upto fc7 layer, and initialize fc8 weights and biases from scratch. Then train the network as before and report the training/validation curves and final performance on PASCAL test set.
Run code: python 04_pascal_vgg16_finetune.py VOCdevkit

## Task 5: Analysis
#### Conv-1 filters
#### Nearest neighbors
#### tSNE visualization of intermediate features
#### Are some classes harder?
Run code: 
python 05_Alex.py VOCdevkit 
python 05_VGG.py VOCdevkit
python Task5_local_Alex.py 
python Task5_local_VGG.py

## Task 6: Improve the classification performance 
Run code: 
python 06_Alex.py VOCdevkit 
python 06_VGG.py VOCdevkit
