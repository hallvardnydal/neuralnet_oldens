# Functions that define a training, valid and test set based on the ODE
# mx'' + cx' + kx = 0
# and generates a clean and noisy time series from this equation.

import numpy as np
import theano

def generate_sample():
    
    # initial condition, constants
    h0 = np.random.uniform(low=0.5,high=1.5)  # initial deviation from equilibrium in m
    v0 = 0.0                                  # m/s
    t0 = 0.0                                  # s

    # constants
    time = 20.           # duration in sec
    dt = 0.01            # time step in sec
    nstep = int(time/dt) # number of time steps

    k = np.random.uniform(low=10,high=15)         # spring constant in N/m
    m = np.random.uniform(low=0.5,high=1.5)       # mass in kg
    k1_drag = np.random.uniform(low=0.2,high=0.5) # drag constant
        
    c1 = k/m
    c2 = k1_drag/m

    # array definitions
    h = np.zeros((nstep,), dtype=np.float32)  # height in meters
    v = np.zeros((nstep,), dtype=np.float32)  # velocity in m/sec
    a = np.zeros((nstep,), dtype=np.float32)  # acceleration in m/sec^2
    t = np.zeros((nstep,), dtype=np.float32)  # time in sec
   
    # initial conditions 
    i = 0
    v[0] = v0
    h[0] = h0
    t[0] = t0

    for i in np.arange(1,nstep):

        a[i-1] = -c1 * h[i-1] - c2* v[i-1]
        v[i] = v[i-1] + a[i-1] * dt
        h[i] = h[i-1] + v[i-1]*dt
        t[i] = t[i-1] + dt
        
    return a,v,h,t,c1,c2
    
def downsample(a,v,h,t,sampling_rate = 10):
    a = a[::sampling_rate]
    v = v[::sampling_rate]
    h = h[::sampling_rate]
    t = t[::sampling_rate]
    return a,v,h,t
    
def normalize(tt_set):
    for n in xrange(tt_set.shape[0]):
        tt_set[n] -= tt_set[n].mean()
        tt_set[n] /= tt_set[n].std()
    return tt_set
    
def gen_data(tot_samples = 1000,val_samples = 200, test_samples = 200,noise=0.1):
    set_x = np.zeros((tot_samples,200))
    set_y = np.zeros((tot_samples,2))
    
    for n in xrange(tot_samples):
        
        # generate sample
        a,v,h,t,c1,c2 = generate_sample()
        a,v,h,t = downsample(a,v,h,t)
        
        # define set
        set_x[n]   = h[:200]
        set_y[n,0] = c1
        set_y[n,1] = c2
        
    # add noise set
    set_x_noise = set_x + np.random.uniform(low=(-noise),high=(+noise),size=set_x.shape)
    
    # normalize set
    set_x = normalize(set_x)
    
    # define training, valid and test sets 
    train_set_x, train_set_y = set_x[:(tot_samples-test_samples)], set_y[:(tot_samples-test_samples)]
    test_set_x,test_set_y = set_x[(tot_samples-test_samples):], set_y[(tot_samples-test_samples):]
    valid_set_x,valid_set_y = test_set_x[:val_samples],test_set_y[:val_samples]
 
    # cast 
    train_set_x,train_set_y = train_set_x.astype(np.float32),train_set_y.astype(np.float32)
    valid_set_x,valid_set_y = valid_set_x.astype(np.float32),valid_set_y.astype(np.float32)
    test_set_x,test_set_y   = test_set_x.astype(np.float32),test_set_y.astype(np.float32)
    
    # define theano shared variables
    train_set_x,train_set_y = theano.shared(train_set_x),theano.shared(train_set_y)
    valid_set_x,valid_set_y = theano.shared(valid_set_x),theano.shared(valid_set_y)
    test_set_x,test_set_y   = theano.shared(test_set_x),theano.shared(test_set_y)   
    
    data = [[train_set_x,train_set_y],[valid_set_x,valid_set_y],[test_set_x,test_set_y]]
    
    return data
    
if __name__ == "__main__":
    data = gen_data()
