"""
Trains a Tensorflow image classifier on MNIST
"""


import tensorflow as tf
import data_setup, model_builder

#Set up hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 128

HIDDEN_UNITS = 128
LEARNING_RATE = 1e-3

#Define preprocessing function
def transform(img, label):
    img = tf.cast(img,tf.float32)/255.0
    img = tf.expand_dims(img, axis = -1)
    #img = tf.expand_dims(img, axis = -1)

    return img, label

#Create datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_dataset, test_dataset = data_setup.create_datasets(x_train, 
                                                         y_train, 
                                                         x_test, 
                                                         y_test, 
                                                         transform = transform,
                                                         batch_size = BATCH_SIZE)


#Create model
model = model_builder.ConvModel([10,10,10],[3,3,3], 10) #ten digits in MNIST
#model = model_builder.ShallowDense(HIDDEN_UNITS, 10) #ten digits in MNIST

#Set loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

#Compile and fit
model.compile(optimizer = optimizer, loss = loss_fn, metrics = ['acc'])

#print(model.summary())

model.fit(train_dataset,
          epochs = NUM_EPOCHS,
          validation_data = test_dataset)

#save model
#model.save('models/dense')
model.save('models/conv')
