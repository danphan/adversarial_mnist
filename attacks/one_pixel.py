"""
Defines the class implementing the one pixel attack:
Paper: https://arxiv.org/abs/1710.08864
"""

from attack import Attack
from scipy.optimize import differential_evolution


class OnePixel(Attack):
    """
    Creates an instance of the DeepFool class. At this time, this only works on a batch of length 1.

    Args
    ----
    estimator : tf.keras.Model
        trained classifier, which we can call on batches, as well as obtain gradients from
    mutation : float
        the mutation rate for differential evolution
    popsize : int
        the number of children generated in differential evolution
    """
    def __init__(self, estimator, mutation = 0.5, popsize = 20):
        super().__init__()
        self.estimator = estimator
        self.mutation = mutation
        self.popsize = popsize
        OnePixel._check_params(self)

    def _check_params(self):
        #check mutation
        if not isinstance(self.mutation, (int, float)):
            raise TypeError('mutation must be a float')
        if self.mutation < 0:
            raise ValueError('mutation must be positive')
        if self.mutation > 2:
            raise ValueError('mutation must be less than 2')

        #check popsize
        if not isinstance(self.popsize, int):
            raise TypeError('popsize must be a positive integer')
        if self.popsize <= 0:
            raise ValueError('popsize must be a positive integer')

    def generate(self, img, target):
        """Given an input img classified by the network as c, returns an output image which is only different in one pixel, and hopefully is not classified as c.

        Currently, this implementation only works with a batch of size 1.

        Args
        ----
        img : Tensorflow Tensor with batch size 1
            input image
        target : Tensor with batch size 1
            class we want the model to classify the modified image as
        """
        prediction = self.estimator(img)
        class_predicted = tf.math.argmax(prediction, axis = 1).numpy()
        if class_predicted == target:
            raise Exception('Image is already classified by model as target.')

        #return probability of target class
        #x and y can take integer values from 0 to 27 (inclusive), while z is a float between 0 and 1

        def func(x):
            img_mod = self._modify_img(img, x)
            pred = self.estimator(img_mod)
            probs =  tf.nn.softmax(pred)
            prob_target = probs[0,target].numpy()
            return 1-prob_target #minus sign because we want to maximize this prob (i.e. minimize 1-prob)

        opt = differential_evolution(func, 
                                     bounds = [(0,img.shape[0]-1),(0,img.shape[1]-1),(0,1)], 
                                     integrality = [True, True, False], 
                                     popsize = self.popsize, 
                                     mutation = self.mutation, 
                                     polish = False)
        print('Success? {}'.format(opt.success))
        print('Reason for termination: {}'.format(opt.message))
        return self._modify_img(img, opt.x)

    def _modify_img(self, img, mod, batched = True):
        x, y, z = mod
        x = int(x)
        y = int(y)
        img_np = img.numpy()
        img_np[0,x,y,0] = z
        return tf.convert_to_tensor(img_np, dtype = tf.float32)


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
    f = OnePixel(model, mutation = 1, popsize = 40)
    img_perturbed = f.generate(img_tensor,3)

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

