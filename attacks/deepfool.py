"""
Defines the class implementing the DeepFool attack:
Paper: https://arxiv.org/abs/1511.04599
"""

from attack import Attack

class DeepFool(Attack):
    """
    Creates an instance of the DeepFool class. At this time, this only works on a batch of length 1.

    Args
    ----
    estimator : tf.keras.Model
        trained classifier, which we can call on batches, as well as obtain gradients from
    eta : float
        the step size to take.
    num_iter : int
        the number of iterations
    """
    def __init__(self, estimator, eta, num_iter):
        super().__init__()
        self.estimator = estimator
        self.eta = eta
        self.num_iter = num_iter
        DeepFool._check_params(self)

    def _check_params(self):
        #check eta
        if not isinstance(self.eta, float):
            raise TypeError('eta must be a float')
        if self.eta < 0:
            raise ValueError('eta must be positive')

        #check num_iter
        if not isinstance(self.num_iter, int):
            raise TypeError('num_iter must be a positive integer')
        if self.num_iter <= 0:
            raise ValueError('num_iter must be a positive integer')

    def generate(self, img):
        """Given an input img classified by the network as c, returns an output image which is not classified as c.

        Args
        ----
        img : Tensorflow Tensor with batch size 1
            input image
        """

        class_0 = tf.math.argmax(self.estimator(img), axis = -1)

        for i in range(self.num_iter):
            print(f'Iteration {i}')
            #calculate og label from model
            with tf.GradientTape() as tape:
                tape.watch(img)
                prediction = self.estimator(img)
            #get class that img was classified as
            class_0 = tf.math.argmax(prediction, axis = -1)

            #get value of f for the predicted class
            f0 = tf.gather(prediction, class_0, axis = 1, batch_dims = 1)

            jacobian = tape.batch_jacobian(prediction, img)
            w0 = tf.gather(jacobian, class_0, axis = 1, batch_dims = 1)

            distances_num = tf.math.abs(prediction - f0[:, tf.newaxis])
            distances_denom = tf.math.reduce_euclidean_norm(jacobian - tf.expand_dims(w0, axis = 1), axis = [-3,-2,-1]) #calculate euclidean norm over the image dimensions 

            distances = tf.math.divide_no_nan(distances_num, distances_denom) #make sure that there aren't any problems with dividing by zero

            #obtain closest class which is not the one predicted by the model
            #the first element in the second axis is to ignore the "distance zero" classes, which are just the classes predicted by the model
            classes_closest = tf.argsort(distances, axis = -1, direction = 'ASCENDING')[:,1] 

            f_closest = tf.gather(prediction, classes_closest, axis = 1, batch_dims = 1) #find value of loss for this closest class
            w_closest = tf.gather(jacobian, classes_closest, axis = 1, batch_dims = 1)   #find gradient for this closest class

            #find the perturbation which gets us to the decision boundary
            r = (1+self.eta)*(tf.math.abs(f_closest-f0)/tf.math.reduce_euclidean_norm(w_closest-w0, axis = [-3,-2,-1])**2 )[:, tf.newaxis, tf.newaxis, tf.newaxis] * (w_closest - w0)
            
            print('norm of modification: {}'.format(tf.norm(r)))

            img += r

            class_predict = tf.math.argmax(self.estimator(img), axis = -1)
            if class_predict != class_0:
                print('Network fooled')
                return img

        print('Network not fooled')
        return img

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    img =  x_train[0] / 255.
    y = y_train[0]
    img_tensor = tf.Variable(img, dtype = tf.float32)
    img_tensor = tf.expand_dims(img_tensor,-1)
    img_tensor = tf.expand_dims(img_tensor,0)


    model = tf.keras.models.load_model('../models/conv')
    f = DeepFool(model, eta = 1e-6, num_iter = 20)
    img_perturbed = f.generate(img_tensor)

    print(f'Actual label: {y}')
    preds = model(img_perturbed)
    probs = tf.nn.softmax(preds, axis = -1)
    class_pred = tf.math.argmax(probs, axis = 1)
    prob_class = probs[:,class_pred.numpy()[0]]
    print(f'Perturbed label: {class_pred} with probability {prob_class}')
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(img_perturbed[0])
    plt.show()

