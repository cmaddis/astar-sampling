"""Draw a sample from a sorted softmax."""

import astar
import numpy as np

class SortedSoftmax(object):
    
    def __init__(self, x, beta):
        self.x = np.sort(x)
        self.beta = beta
        self.logz = astar.logsumexp(self.x) 
        
    def dimensions(self):
        return 1
        
    def negative_energy(self, sample):
        return self.x[sample]
    
    def state_space(self):
        return (0, len(self.x))

class Uniform(object):
            
    def log_volume(self, subset):
        left, right = subset
        return np.log(right-left)

    def negative_energy(self, sample):
        return 0
        
    def sample(self, subset):
        left, right = subset
        sample = np.random.randint(right-left) + left
        return sample

def bounder(distribution, proposal, subset):
    left, right = subset
    return distribution.negative_energy(right - 1)

def splitter(subset, sample):
    left, right = subset
    
    if right - left > 1:
        if sample == right - 1:
            return [(left, sample), (sample, sample+1)]
        elif left == sample:
            return [(sample, sample + 1), (sample + 1, right)]
        else:
            return [(left, sample), (sample, sample+1), (sample + 1, right)]
    else:
        return [(left, right)]


    
def main():
    x = np.array([1, 1, -1, 0.25, 0.25])
    target = SortedSoftmax(x, 1)
    proposal = Uniform()
    M = 50000
    samples, gumbels = np.zeros(M), np.zeros(M)
    stream = astar.astar_sampling_iterator(target, proposal, bounder, splitter)
    for j in range(M):
        X, G = stream.next()
        samples[j] = X
        gumbels[j] = G
    
    # post process
    untruncated_gumbels = astar.untruncate(gumbels[1:], gumbels[:-1])
    gumbels = np.concatenate(([gumbels[0]], untruncated_gumbels))
    empirical_distribution = np.zeros((M, len(x)))
    empirical_distribution[np.arange(M), samples.astype(int)] = 1

    print("true distribution:{0}".format(np.exp(target.x - target.logz)))
    print("empirical distribution:{0}".format(np.mean(empirical_distribution, axis=0)))
    print("logz:{0}, hat(logz):{1}".format(target.logz, gumbels.mean() - np.euler_gamma))
    
if __name__ == '__main__':
    main()