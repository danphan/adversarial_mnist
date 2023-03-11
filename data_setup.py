import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
#import time
#import random

def create_datasets(x_train, y_train, x_test, y_test, transform, batch_size):
    """Creates training and testing datasets.

    Takes in the training and testing instances and turns
    them into Tensorflow Datasets.

    Args
    ------------
    x_train : ndarray
        Training images
    y_train : ndarray
        Training labels
    x_test : ndarray
        Test images
    y_test : ndarray
        Test labels
    transform: callable
        Function for preprocessing each instance (x,y) ---> (x_processed, y_processed).
    batch_size: int
        Integer representing the number of instances per batch.

    Returns
    -------------
    tuple
        (train_dataset, test_dataset).
    """

    def create_dataset(x,y,transform, batch_size):
        assert len(x) == len(y)

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.map(transform,
          num_parallel_calls=tf.data.AUTOTUNE) #apply normalize_img to all of the data
        ds = ds.cache() #store data in cache so that it can be loaded faster
        ds = ds.shuffle(len(x)) #make sure to shuffle data for randomness. tell shuffle method the size of the dataset to ensure randomness
        ds = ds.batch(batch_size) #make batch size power of 2 for efficiency
        ds = ds.prefetch(tf.data.AUTOTUNE) #preprocess next batch while performing current training step

        return ds

    ds_train = create_dataset(x_train, y_train, transform, batch_size)
    ds_test = create_dataset(x_test, y_test, transform, batch_size)

    return ds_train, ds_test



if __name__ == '__main__':

    #load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )

    transform = lambda img, label: (tf.cast(img, tf.float32)/255., label)
    batch_size = 128

    train_dataset, test_dataset = create_datasets(x_train,
                                                  y_train,
                                                  x_test,
                                                  y_test,
                                                  transform,
                                                  batch_size)


