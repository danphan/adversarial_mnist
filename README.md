# Adversarial MNIST

I implement the Fast Gradient Sign Method (FGSM) by [Goodfellow et al.](https://arxiv.org/abs/1412.6572), Projected Gradient Descent (PGD) by [Madry et al.](https://arxiv.org/abs/1706.06083), DeepFool by [Moosavi-Dezfooli et al.](https://arxiv.org/abs/1511.04599), and the One-Pixel attack by [Su et al.](https://arxiv.org/abs/1710.08864).
(In the future, I also plan to implement the adversarial patch method.)

In this package, there are two models which have been trained on MNIST:
1. A shallow fully-connected neural network, with 1 hidden layer.
2. A ResNet-type network, with 3 residual convolutional blocks, and 1 fully-connected layer at the end.

The trained models can be found in the models directory.

Below is an example of the adversarial attacks on an image:

## Example of the implemented attacks
![Generated adversarial images](attack_pic.png)

All attacks can be found in the attacks directory, and trainers to fit on adversarial examples can be found in the defenses directory.

The examples.ipynb notebook shows how the code can be used.

I then fine-tune this neural network on adversarial examples to yield an adversarially robust neural network. As suggested in [Jeddi et al.](https://arxiv.org/abs/2012.13628), I use a learning rate schedule which gradually warms up, and then cools down.
They suggest that this yields a neural network which does not overfit on the adversarial examples, and forgets the distribution of natural images. In the simple case of MNIST, I find that this learning rate schedule does not make a difference, and even a fixed learning rate yields a robust network.

