"""
Defines the class implementing the fast gradient sign method, from https://arxiv.org/abs/1412.6572
"""

from attack import Attack

class FGSM(Attack):
    """
    Create an instance of the Fast Gradient Sign Method class.

    Args
    ----
    estimator : tf.keras.Model
        trained classifier, which we can call on batches, as well as obtain gradients from
    eps : float
        epsilon-- the step size to take.
    """
    def __init__(self, estimator, eps):
        super().__init__()
        self.estimator = estimator
        self.eps = eps
        FGSM._check_params(self)

    def _check_params(self):
        if not isinstance(self.eps, float):
            raise TypeError('eps must be a float')
        if self.eps < 0:
            raise ValueError('eps must be positive')

    def generate(self, x, y):
        """Generate adversarial images given input images x, and their true classes y.
        
        Args
        ----
        x : Tensorflow Tensor or numpy array
            batch of input images
        y : Tensorflow Tensor or numpy array
            batch of ground truths
        """

        #calculate gradient of loss with respect to images
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = self.estimator(x)
            #tensorflow implementation of compute_loss doesn't use x
            loss = self.estimator.compute_loss(y = y, y_pred = pred) 

        gradient = tape.gradient(loss, x)

        return x + self.eps * tf.sign(gradient)


if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    img =  x_train[0] / 255.
    y = y_train[0]
    y = tf.Variable(y, dtype = tf.float32)
    y = tf.expand_dims(y, axis = 0)
    img_tensor = tf.Variable(img, dtype = tf.float32)
    img_tensor = tf.expand_dims(img_tensor,-1)
    img_tensor = tf.expand_dims(img_tensor,0)


    model = tf.keras.models.load_model('../models/conv')
    f = FGSM(model, eps = 1e-1)
    img_perturbed = f.generate(img_tensor, y)

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

