"""
Base class for evasion attacks.
"""

class Attack:
    def generate(self, x, **kwargs):
        """Generate adversarial examples, given inputs x. This method needs to be defined in every subclass of Attack.

        Args
        ----
        x : tf.Tensor
            batch of examples for evasion.

        Returns
        -------
        x_perturbed : tf.Tensor
           batch of perturbed examples that the model hopefully misclassifies. 
        """
        raise NotImplementedError
