"""Draw a sample from p(x) \propto exp(sin(x)) over [0, 2pi]"""

import matplotlib.pyplot as plt
import astar
import osstar
import numpy as np
from scipy.integrate import quad
pi = np.pi

class Sin(object):
    
    def __init__(self, beta):
        self.beta = beta
        self.zstore = None
        
    def dimensions(self):
        return 1
        
    def negative_energy(self, sample):
        return self.beta*np.sin(sample)
    
    def state_space(self):
        return (0, 2*pi)
    
    def log_density(self, sample):
        return self.negative_energy(sample) - np.log(self.z())
        
    def z(self):
        if self.zstore is not None:
            return self.zstore
        else:
            left, right = self.state_space()
            self.zstore, err = quad(lambda x: float(np.exp(self.negative_energy(x))), left, right)
            return self.zstore
    
class Uniform(object):

    def __init__(self):
        self.counter = 0
   
    def log_volume(self, subset):
        left, right = subset
        return np.log(right-left)

    def negative_energy(self, sample):
        return 0
        
    def sample(self, subset):
        left, right = subset
        self.counter += 1
        return np.random.rand()*(right-left) + left


class Bounder(object):
    
    def __init__(self):
        self.counter = 0

    def __call__(self, distribution, proposal, subset):
        left, right = subset
        self.counter += 1
        if left <= pi * 0.5 <= right:
            return distribution.negative_energy(pi * 0.5)
                
        elif distribution.negative_energy(left) > distribution.negative_energy(right):
            return distribution.negative_energy(left)
    
        else:
            return distribution.negative_energy(right)

class Splitter(object):
    
    def __init__(self):
        self.counter = 0
    
    def __call__(self, subset, sample):
        left, right = subset
        self.counter += 1
        return [(left, sample), (sample, right)]


def main():
    import time
    target = Sin(1)
    proposal = Uniform()
    bounder = Bounder()
    splitter = Splitter()
    M = 1
    N = 5000

    rej_samples = []    
    print("\nM samples per run:{0}, N runs:{1}".format(M, N))

    start = time.time()
    for i in range(N):
        stream = osstar.rejection_sampling_iterator(target, proposal, bounder)
        for j in range(M):
            X, G = stream.next()
            rej_samples.append(X)
    end = time.time()
    rej_samples = np.array(rej_samples).ravel()

    print("\nRejection Sampling")
    print("likelihoods/sample: {0}".format(proposal.counter/float(M*N)))
    print("bounds/sample: {0}".format(bounder.counter/float(M*N)))
    print("time taken: {0}s".format(end-start))
    splitter.counter = 0
    proposal.counter = 0
    bounder.counter = 0


    osstar_samples = []    
    start = time.time()
    for i in range(N):
        stream = osstar.osstar_sampling_iterator(target, proposal, bounder, splitter)
        for j in range(M):
            X, G = stream.next()
            osstar_samples.append(X)
    end = time.time()
    osstar_samples = np.array(osstar_samples).ravel()

    print("\nOS*")
    print("splits/run: {0}".format(splitter.counter/float(N)))
    print("likelihoods/sample: {0}".format(proposal.counter/float(M*N)))
    print("bounds/sample: {0}".format(bounder.counter/float(M*N)))
    print("time taken: {0}s".format(end-start))
    splitter.counter = 0
    proposal.counter = 0
    bounder.counter = 0

    astar_iter_samples = []    
    start = time.time()
    for i in range(N):
        stream = astar.astar_sampling_iterator(target, proposal, bounder, splitter)
        for j in range(M):
            X, G = stream.next()
            astar_iter_samples.append(X)
    end = time.time()
    astar_iter_samples = np.array(astar_iter_samples).ravel()

    print("\nA*")
    print("splits/run: {0}".format(splitter.counter/float(N)))
    print("likelihoods/sample: {0}".format(proposal.counter/float(M*N)))
    print("bounds/sample: {0}".format(bounder.counter/float(M*N)))
    print("time taken: {0}s".format(end-start))
    
    plt.subplot(221)
    plt.hist(rej_samples, bins=25, normed=True)
    plt.title("Rejection")
    p = np.vectorize(lambda x: np.exp(target.log_density(x)))
    x = np.linspace(0, 2*pi, 1000)
    y = p(x)
    plt.plot(x,y, "r", linewidth=4)
    
    plt.subplot(222)
    plt.hist(osstar_samples, bins=25, normed=True)
    plt.title("OS*")
    p = np.vectorize(lambda x: np.exp(target.log_density(x)))
    x = np.linspace(0, 2*pi, 1000)
    y = p(x)
    plt.plot(x,y, "r", linewidth=4)

    plt.subplot(223)
    plt.hist(astar_iter_samples, bins=25, normed=True)
    plt.title("A*")
    p = np.vectorize(lambda x: np.exp(target.log_density(x)))
    x = np.linspace(0, 2*pi, 1000)
    y = p(x)
    plt.plot(x,y, "r", linewidth=4)


    plt.draw()
    plt.show()
    
    
if __name__ == '__main__':
    main()