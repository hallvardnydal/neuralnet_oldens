
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams

from generate_data import gen_data
from mlp           import MLP
    
def test_mlp(learning_rate=0.01, 
            n_epochs=100, 
            batch_size=20, 
            n_hidden=200):

    datasets = gen_data(noise=0.2)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    index = T.lscalar() 
    x = T.matrix('x')  
    y = T.matrix('y')  

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        input_dim = (batch_size,200),
        n_in= train_set_x.get_value().shape[1],
        n_hidden=n_hidden,
        n_out=2
    )

    classifier_test = classifier.TestVersion(
        rng=rng,
        input=x,
        input_dim = (batch_size,200),
        n_in= train_set_x.get_value().shape[1],
        n_hidden=n_hidden,
        n_out=2,
        classifier = classifier
    )

    cost = classifier.L2(y)

    # compiling Theano functions
    
    # Calculates the output of the test data
    test_model = theano.function(
        inputs=[index],
        outputs=classifier_test.last_layer.output,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size]}
    )
    
    # Calculates the output of the hidden layer
    #hidden_model = theano.function(
    #    inputs=[index],
    #    outputs=classifier_test.hiddenLayer.output,
    #    givens={
    #        x: test_set_x[index * batch_size:(index + 1) * batch_size]}
    #)
    
    # Calculate the L2 error of the validation set
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier_test.L2(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # Gradient and update function
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    # Init train model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    for n in xrange(n_epochs):
        costs = [train_model(i) for i in xrange(n_train_batches)]
        val_costs = [validate_model(i) for i in xrange(n_valid_batches)]
        print "Cost:",np.mean(costs),"Val error:",np.mean(val_costs)
    
    # Calculate outout of test set
    output = np.zeros((0,2))
    for i in xrange(n_test_batches):
        output = np.vstack((output,test_model(i)))
    
    #hidden = np.zeros((0,200))
    #for i in xrange(n_test_batches):
     #   hidden = np.vstack((hidden,hidden_model(i)))
     
    print "... finished training"
     
    print " "
    print "Mean value, a:",np.mean(output[:,0])
    print "Mean L2-error, a:",np.mean((output[:,0]-test_set_y.get_value()[:,0])**2)
    print "Mean value, b:",np.mean(output[:,1])
    print "Mean L2-error, b:",np.mean((output[:,1]-test_set_y.get_value()[:,1])**2)
    
    ########################
    ##   Olden's method   ##
    ########################
    
    # Calculate the D matrix 
    D = np.dot(classifier.params[0].get_value(),classifier.params[2].get_value())
    
    # Calculate the absolute value of the D-matrix
    cw    = np.abs(D)
    
    # Normalize
    for n in xrange(cw.shape[1]):
        cw[:,n]    /= cw[:,n].sum()
    test_x = datasets[0][0].get_value()


    f, axarr = plt.subplots(4, sharex=True)
    f.tight_layout() 
    axarr[0].plot(test_x[0],label="input")
    axarr[0].set_title('Example time series, x (m)')
    axarr[1].plot(cw[:,0],color="red") 
    axarr[1].set_title('Relative contribution, oscillation term $a$ ')
    axarr[2].plot(cw[:,1],color="blue")
    axarr[2].set_title('Relative contribution, friction term $b$')
    axarr[3].plot(np.cumsum(cw[:,0]),color="red")
    axarr[3].plot(np.cumsum(cw[:,1]),color="blue")
    axarr[3].set_title('Cummulative contribution, $a$ (red) and $b$ (blue)')
    
    plt.show()
    
    
#def adjust(matrix,hidden,thres=10**(-14)):
#    
#    for n in xrange(hidden.shape[0]):
#        count = 0
#        for m in xrange(hidden.shape[1]):
#            if hidden[n,m] < thres:
#                count += 1
#        matrix[n] /= (1-(count/float(hidden.shape[1])))
#    
#    return matrix

    
if __name__ == '__main__':
    test_mlp()
