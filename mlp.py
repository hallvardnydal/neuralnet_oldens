import theano
import theano.tensor as T

from hidden import HiddenLayer

class MLP(object):

    def __init__(self, rng, input,input_dim, n_in, n_hidden, n_out, test = False,  classifier = None, dropout_p=0.5):

        if test == False:
            input = input.reshape(input_dim)
            
            # Initialize hidden layer
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
            )
            
            # Initialize output layer
            self.last_layer = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out
            )

            # save parameters
            self.params = self.hiddenLayer.params + self.last_layer.params

        else:
            input            = input.reshape(input_dim)
            self.hiddenLayer = classifier.hiddenLayer.TestVersion(rng, input, n_in, n_out)
            self.last_layer  = classifier.last_layer.TestVersion(rng,
                                                                input = self.hiddenLayer.output,
                                                                n_in = n_hidden,
                                                                n_out = n_out)

    def L2(self,y):
        """
        Calculates the L2 norm between the output
        of the last layer and the ground truth.
        """
        return T.mean((self.last_layer.output-y)**2)

    def TestVersion(self,rng, input, input_dim, n_in, n_hidden, n_out, classifier):
        return MLP(rng, input, input_dim, n_in, n_hidden, n_out, test = True, classifier = classifier)