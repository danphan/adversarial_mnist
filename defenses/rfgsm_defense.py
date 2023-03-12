"""
Implementation of the randomly-initialized FGSM adversarial training method.
Paper: https://arxiv.org/abs/2001.03994
"""
from attacks import fgsm, pgd
import tensorflow as tf


class RFGSM:
    """Creates an instance of a class which adversarially trains a model using the RFGSM.

    Args
    ----
    classifier: tf.keras.Model
        the model to be trained adversarially.
    eps: float
        the maximum size of the perturbation one can take
    """

    def __init__(self, classifier, eps):
        self.classifier = classifier
        self.eps = eps
        super().__init__()

    def fit(self, train_dataset, validation_dataset = None, optimizer = None, num_epochs = 10):
        """
        Adversarially trains the model.

        Args
        ----
        train_dataset: tf.data.Dataset
            Training dataset.
        validation_dataset: tf.data.Dataset
            Validation dataset
        num_epochs: int
            Number of epochs to train for.
        """

        if optimizer:
            self.classifier.compile(optimizer = optimizer,
                                    loss = self.classifier.loss,
                                    metrics = self.classifier.metrics)

        
        attacker = fgsm.FGSM(self.classifier, self.eps)
        for idx in range(num_epochs):
            print(f'Epoch {idx}')

            #loop through training data
            for batch in train_dataset:
                x, y = batch

                #randomly initialize perturbation
                delta = tf.random.uniform(x.shape, -self.eps, self.eps)
                x_mod = x + delta

                x_adv = attacker.generate(x_mod, y)
                x_adv = x + tf.clip_by_value(x_adv-x, -self.eps, self.eps)
#                #calculate gradient
#                with tf.GradientTape() as tape:
#                    tape.watch(x_mod)
#                    pred = self.classifier(x_mod)
#                    loss_fn = self.classifier.compute_loss(y_pred = pred, y = y)
#                grad = tape.gradient(loss_fn, x_mod)
#
#                #calculate perturbation
#                delta += self.eps * tf.sign(grad)
#
#                #clip perturbation
#                delta = tf.clip_by_value(delta, -self.eps, self.eps)
#
#                #calculate adversarial example
#                x_adv = x + delta

                #perform gradient descent
                with tf.GradientTape() as tape:
                    y_pred = self.classifier(x_adv, training = True)
                    loss_fn = self.classifier.compute_loss(y_pred = y_pred, y = y)
                grads = tape.gradient(loss_fn, self.classifier.trainable_weights)
                
                #update weights
                self.classifier.optimizer.apply_gradients(zip(grads, self.classifier.trainable_weights))

            #calculate loss and accuracy on validation dataset
            print('Calculating validation metrics')
            epoch_loss_val = tf.keras.metrics.Mean()
            epoch_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()
            
            adv_epoch_loss_val = tf.keras.metrics.Mean()
            adv_epoch_acc_val = tf.keras.metrics.SparseCategoricalAccuracy()

            attacker = pgd.PGD(self.classifier, eta = self.eps/5, eps = self.eps, num_iter = 10)
            for batch in validation_dataset:
                x, y = batch

                loss_val = self.classifier.compute_loss(y_pred = self.classifier(x), y = y)
                epoch_loss_val.update_state(loss_val)
                epoch_acc_val.update_state(y, self.classifier(x))

                x_adv = attacker.generate(x,y)
                adv_loss_val = self.classifier.compute_loss(y_pred = self.classifier(x_adv), y = y)
                adv_epoch_loss_val.update_state(adv_loss_val)
                adv_epoch_acc_val.update_state(y, self.classifier(x_adv))


            print('validation loss: {}'.format(epoch_loss_val.result()))
            print('validation accuracy: {}'.format(epoch_acc_val.result()))
            print('validation adversarial loss: {}'.format(adv_epoch_loss_val.result()))
            print('validation adversarial accuracy: {}'.format(adv_epoch_acc_val.result()))

                
               


if __name__ == '__main__':
    import data_setup

    model = tf.keras.models.load_model('../models/dense')
    trainer = RFGSM(model,1e-1)


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    transform = lambda img, label: (tf.expand_dims(tf.cast(img, tf.float32), axis = -1)/255., label)
    batch_size = 128
    train_dataset, test_dataset = data_setup.create_datasets(x_train, 
                                                             y_train, 
                                                             x_test, 
                                                             y_test,
                                                             transform,
                                                             batch_size)


    trainer.fit(train_dataset, test_dataset)

