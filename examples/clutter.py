"""Sample from the clutter problem posterior [1].

Consider the following model

theta ~ N(0, sigma0)
for i in range(n):
  xi ~ pi N(theta, 1) + (1-pi) N(0, 100)

So each data point xi is generated as a mixture between an outlier distribution 
and a fairly tight Gaussian. Now the log posterior is up to an additive constant

log N(theta; 0, sigma0) + sum log (pi N(xi; theta, 1) + (1-pi) N(xi; 0, 100))

In this code we draw sample from the posterior using a variety of sampling techniques.
All techniques require a proposal (we use the prior) and bounds on the likelihood
restricted to hypercubes. We independently upper bound each term of the likelihood sum.
So even though the full posterior can be multimodal, each term in the sum is unimodal 
as a function of theta, and we can compute a total upper bound just as a sum of individual 
upper bounds.

[1] Thomas P. Minka. Expectation Propagation for Approximate Bayesian Inference.
In UAI, pages 361-369. Morgan Kaufmann Publishers Inc., 2001.
"""


import matplotlib.pyplot as plt
import osstar
import numpy as np
import astar
from scipy.special import erf
from scipy.stats import truncnorm, truncexpon
from scipy.integrate import quad

def logaddexp(loga, logb):
    '''Compute log(exp(loga) + exp(logb)) while minimizing the possibility of
    over/underflow.'''
    
    m = np.maximum(loga, logb)
    return np.log(np.exp(loga - m) + np.exp(logb - m)) + m

class Clutter(object):
    """
    The clutter problem model.
    """
    
    def __init__(self, sigma, pi):
        assert isinstance(sigma, np.ndarray)        
        self.pi = pi
        self.sigma = sigma        
        self.dim = len(sigma)
        
    def sample(self, m):
        theta = npr.randn(self.dim)*self.sigma
        x = []
        for i in range(m):
            
            if npr.rand() < self.pi:
                x.append(theta + npr.randn(self.dim))
            else:
                x.append(npr.randn(self.dim)*100)
        
        return theta, np.array(x)
        
    def posterior(self, data):
        assert data.shape[1] == (self.dim)
        return ClutterPosterior(self.sigma, self.pi, data)


class ClutterPosterior(object):
    """
    A distribution object for the posterior of the clutter problem.
    """
    
    
    def __init__(self, sigma, pi, data):
        
        self.pi = pi
        self.data = data
        self.num_points, self.dim = data.shape
        self.sigma = sigma
        self.zstore = None
    
    def dimensions(self):
        return self.dim
        
    def state_space(self):
        return (-np.inf*np.ones(self.dim), np.inf*np.ones(self.dim))

    def log_prior(self, theta):
        return -0.5*(theta ** 2/self.sigma**2).sum() - self.dim*0.5*np.log(2*np.pi) - np.log(self.sigma).sum()

    def log_likelihoods(self, theta):
        e = 0
        for i in range(self.num_points):
            e += self.log_likelihood(theta, i)
        return e

    def log_likelihood(self, theta, i):
        model = -0.5*((self.data[i,:] - theta) ** 2).sum() - self.dim * 0.5 * np.log(2*np.pi)
        noise = -0.5*(self.data[i,:] ** 2 / 100. ** 2).sum() - self.dim * 0.5 * np.log(2*np.pi) - self.dim * np.log(100)
        return float(logaddexp(model + np.log(self.pi), noise + np.log(1-self.pi)))

    def negative_energy(self, theta):
        if self.dim == 1 and isinstance(theta, float):
            theta = np.array([theta])
        prior = self.log_prior(theta) 
        likelihoods = self.log_likelihoods(theta)
        return float(prior + likelihoods)
        
    def log_density(self, theta):
        return self.negative_energy(theta) - np.log(self.z())

    def z(self):
        if self.zstore is not None:
            return self.zstore
        elif self.dim == 1:
            self.zstore, err = quad(lambda theta: float(np.exp(self.negative_energy(np.array([theta])))), -np.inf, np.inf)
            return self.zstore
        else:
            raise Exception("Can only compute partition function for 1D.")

class IsotropicGaussian(object):
    """
    Isotropic Gaussian.
    """
    
    def __init__(self, dim, sigma, mu=None):
        self.dim = dim
        self.sigma = sigma if len(sigma) == dim else np.ones(dim)*sigma
        self.mu = np.zeros(dim) if mu is None else mu
        self.counter = 0
        
    def log_volume(self, box):
        bottomcorner, topcorner = box        
        assert len(bottomcorner) == len(topcorner)
        assert len(bottomcorner) == self.dim
        top = 0.5*(1 + erf((topcorner-self.mu)/(np.sqrt(2)*self.sigma)))
        bottom = 0.5*(1 + erf((bottomcorner-self.mu)/(np.sqrt(2)*self.sigma)))   
        return np.log(top - bottom).sum()
        
    def negative_energy(self, state):
        return -0.5*((state-self.mu) ** 2/self.sigma**2).sum() - self.dim*0.5*np.log(2*np.pi) - np.log(self.sigma).sum()

    def sample(self, box):
        bottomcorner, topcorner = box
        assert len(bottomcorner) == len(topcorner)
        assert len(bottomcorner) == self.dim  
        self.counter += 1   
        s = np.empty(self.dim)
        for dim in range(self.dim):
            assert bottomcorner[dim] < topcorner[dim], "({0}, {1})".format(bottomcorner, topcorner)
            s[dim] = truncnorm.rvs((bottomcorner[dim]-self.mu[dim])/self.sigma[dim], (topcorner[dim]-self.mu[dim])/self.sigma[dim], loc=self.mu[dim], scale=self.sigma[dim])
        return s        

class Bounder(object):
    
    def __init__(self):
        self.counter = 0

    def __call__(self, distribution, proposal, box):
        bottom, top = box
        total = 0
        self.counter += 1

        for i in range(distribution.num_points):
    
            point = distribution.data[i,:]
            argmax = np.zeros(len(point))

            for d in range(len(point)):   

                if point[d] < bottom[d]:
                    argmax[d] = bottom[d]
            
                elif point[d] > top[d]:
                    argmax[d] = top[d]
            
                else:
                    argmax[d] = point[d]
    
            total += distribution.log_likelihood(argmax, i)
    
        return float(total)

class Splitter(object):
    
    def __init__(self):
        self.counter = 0
    
    def __call__(self, box, max_gumbel):
        bottomcorner, topcorner = box

        dim = bottomcorner.shape[0]

        split_dim = np.argmax(topcorner - bottomcorner)

        bottom1 = bottomcorner.copy()
        top1 = topcorner.copy()

        bottom2 = bottomcorner.copy()
        top2 = topcorner.copy()

        top1[split_dim] = max_gumbel[split_dim]
        bottom2[split_dim] = max_gumbel[split_dim]
        self.counter += 1
        return [(bottom1, top1), (bottom2, top2)]


def main():
    import time
    # bounder
    bounder = Bounder()

    # splitter
    splitter = Splitter()

    dim = 1
    sigma = 2 * np.ones(dim)
    
    # proposal
    proposal = IsotropicGaussian(dim, sigma)

    # target
    pi = 0.5
    num_points = 3
    points = np.concatenate((np.linspace(-5, -3, num_points), np.linspace(2, 4, num_points)))
    data = np.zeros((len(points), dim))
    for d in range(dim):
        data[:,d] = points
    target = ClutterPosterior(sigma, pi, data)

    # go time!
    M = 1
    N = 500
    print("\nM samples per run:{0}, N runs:{1}".format(M, N))

    start = time.time()
    for i in range(N):
        stream = osstar.osstar_sampling_iterator(target, proposal, bounder, splitter)
        for j in range(M):
            X, G = stream.next()
    end = time.time()

    print("\nOS*")
    print("splits/run: {0}".format(splitter.counter/float(N)))
    print("likelihoods/sample: {0}".format(proposal.counter/float(M*N)))
    print("bounds/sample: {0}".format(bounder.counter/float(M*N)))
    print("time taken: {0}s".format(end-start))
    splitter.counter = 0
    proposal.counter = 0
    bounder.counter = 0
    samples = np.empty((N*M, dim)).squeeze()
    start = time.time()
    for i in range(N):
        stream = astar.astar_sampling_iterator(target, proposal, bounder, splitter)
        for j in range(M):
            X, G = stream.next()
            samples[i*M + j] = X
    end = time.time()

    print("\nA*")
    print("splits/run: {0}".format(splitter.counter/float(N)))
    print("likelihoods/sample: {0}".format(proposal.counter/float(M*N)))
    print("bounds/sample: {0}".format(bounder.counter/float(M*N)))
    print("time taken: {0}s".format(end-start))

    if dim == 1:
        samples = samples.ravel()
        plt.hist(samples, bins=25, normed=True)
        p = np.vectorize(lambda x: np.exp(target.log_density(x)))
        x = np.linspace(-5, 5, 1000)
        y = p(x)
        plt.plot(x,y, "r", linewidth=4)
        plt.draw()
        plt.show()
    if dim == 2:
        plt.plot(samples[:,0], samples[:,1], '.')
        plt.draw()
        plt.show()
    
if __name__ == '__main__':
    main()