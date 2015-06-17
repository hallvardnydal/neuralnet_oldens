import theano
import theano.tensor as T
import numpy as np

from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
    """
    Hidden layer class
    """
    
    def __init__(self, 
                rng, 
                input, 
                n_in, 
                n_out, 
                W=None, 
                b=None,
                dropout_p = 0.5):

        self.srng = RandomStreams(seed=234)
        self.dropout_p = dropout_p
        self.input = input
        
        # Initialize weights
        name = "W_hidden"
        if W is None :
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(2. / (n_in + n_out)),
                    high=np.sqrt(2. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name=name, borrow=True)

        # Initialize bias
        name = "b_hidden"
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name, borrow=True)

        self.W = W
        self.b = b

        self.output = self.ReLU(T.dot(input,self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]


    def ReLU(self,X):
        '''
        Rectified linear unit
        '''
        return T.maximum(X,0)

    def dropout(self,X):                                                  
        '''                                                                     
        Dropout with probability p                                      
        '''                                                                     
        if self.dropout_p>0:                                                                 
            retain_prob = 1-self.dropout_p                                                   
            X *= self.srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
            X /= retain_prob                                                    
        return X 

    def TestVersion(self,rng,input,n_in,n_out):
        return HiddenLayer(rng, input, n_in, n_out, W=self.W, b=self.b,dropout_p = 0.0)