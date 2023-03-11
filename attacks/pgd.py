"""
Defines the class implementing the Projected Gradient Descent Attack.
Paper: https://arxiv.org/abs/1706.06083
"""

from attack import Attack

class PGD(Attack):
    """
    Create an instance of the Projected Gradient Descent class.

    Args
    ----
    estimator : tf.keras.Model
        trained classifier, which we can call on batches, as well as obtain gradients from
    eta : float
        the step size to take.
    eps : float
        the maximum difference a pixel's value can change.
    num_iter : int
        the number of iterations
    """
    def __init__(self, estimator, eta, eps, num_iter):
        super().__init__()
        self.estimator = estimator
        self.eta = eta
        self.eps = eps
        self.num_iter = num_iter
        PGD._check_params(self)

    def _check_params(self):
        #check eps
        if not isinstance(self.eps, float):
            raise TypeError('eps must be a float')
        if self.eps < 0:
            raise ValueError('eps must be positive')

        #check eta
        if not isinstance(self.eta, float):
            raise TypeError('eta must be a float')
        if self.eta < 0:
            raise ValueError('eta must be positive')
        if self.eta > self.eps:
            raise ValueError('eta should be smaller than eps')

        #check num_iter
        if not isinstance(self.num_iter, int):
            raise TypeError('num_iter must be a positive integer')
        if self.num_iter <= 0:
            raise ValueError('num_iter must be a positive integer')

    def generate(self, x, y):
        """Generate adversarial images given input images x, and their true classes y.
        
        Args
        ----
        x : Tensorflow Tensor or numpy array
            batch of input images
        y : Tensorflow Tensor or numpy array
            batch of ground truths
        """

        dx = tf.random.uniform(tf.shape(x), -self.eps, self.eps)
        x_mod = x + dx
        for idx in range(self.num_iter):
            print(f'Iteration: {idx}')
            #calculate gradient of loss with respect to images
            with tf.GradientTape() as tape:
                tape.watch(x_mod)
                pred = self.estimator(x_mod)
                #tensorflow implementation of compute_loss doesn't use x
                loss = self.estimator.compute_loss(y = y, y_pred = pred) 

            gradient = tape.gradient(loss, x_mod)

            #calculate perturbation to img
            dx += self.eps * tf.sign(gradient)
            dx = tf.clip_by_value(dx, -self.eps, self.eps)

            #update img
            x_mod = x + dx


        return x_mod 


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
    f = PGD(model, eta = 3e-3, eps = 3e-2, num_iter = 20)
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

