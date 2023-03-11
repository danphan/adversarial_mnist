import tensorflow as tf


class ShallowDense(tf.keras.Model):
    """Creates a dense neural network with one hidden layer.

    Args
    ----
        hidden_units : int
            The number of neurons in the hidden layer
        output_shape : int
            The number of neurons in the final layer. This should be equal to the number of classes in the case of classification.    
    """

    def __init__(self, hidden_units, output_shape):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(output_shape)

    def call(self, x):
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class ResUnit(tf.keras.layers.Layer):
    """Creates a residual convolutional block.

    Args
    ----
    kernel_size: int
        size of the convolutional filter
    num_filters: int
        number of output feature maps
    strides: int
        size of stride to take
    padding: str (either 'valid' or 'same')
        type of padding for the image, prior to convolution

    """
    def __init__(self, num_filters, kernel_size, strides = 1, padding = 'valid'):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(num_filters, 
                                           kernel_size = kernel_size, 
                                           strides = strides, 
                                           padding = padding)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        conv_out = self.conv(x)
        return self.relu(conv_out) + conv_out



class ConvModel(tf.keras.Model):
    """Creates a convolutional neural network with 3 convolutional layers, 
    and one fully-connected layer at the end for classification.

    Args
    ----
        
        filter_sizes : List(int) of length 3
            list of filter sizes for the 3 convolutional layers

        num_filters : List(int) of length 3
            number of feature maps output by each convolutional lyaer

        hidden_units : number of neurons in the fully-connected layer
    """
    def __init__(self, num_filters, kernel_sizes, hidden_units):
        super().__init__()
        self.conv1 = ResUnit(num_filters[0], kernel_sizes[0])
        self.conv2 = ResUnit(num_filters[1], kernel_sizes[1])
        self.conv3 = ResUnit(num_filters[2], kernel_sizes[2])
        self.dense = tf.keras.layers.Dense(hidden_units)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return self.dense(tf.keras.layers.Flatten()(x))


if __name__ == '__main__':
    import numpy as np

    #model = ConvModel([10,10,10],[3,3,3],10)
    #model = ShallowDense(100, 10)
    img = np.ones((10,28,28,1))
    print(model(img).shape)
