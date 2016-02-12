import matplotlib.pyplot as plt
import osstar
import numpy as np
import astar
from scipy.special import erf
from scipy.stats import truncnorm, truncexpon
from scipy.integrate import quad

class CauchyRegression(object):
    
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.num_points = x.size
        self.sigma = sigma
        self.dim = 1
        self.zstore = None

    def log_prior(self, state):
        return -0.5*((state) ** 2/self.sigma**2) - 0.5*np.log(2*np.pi) - np.log(self.sigma)

    def log_likelihood(self, state, i):
        return -np.log(1 + (self.y[i] - self.x[i]*state)**2)

    def negative_energy(self, state):
        neg_energy = self.log_prior(state)
        for i in range(self.num_points):
            neg_energy += self.log_likelihood(state, i)
        return neg_energy
    
    def state_space(self):
        return (-np.inf*np.ones(self.dim), np.inf*np.ones(self.dim))

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
        # print self.counter
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
    
            point = distribution.x[i]
            target = distribution.y[i]
            mode = target/point
            
            if mode < bottom:
                argmax = bottom
            elif mode > top:
                argmax = top
            else:
                argmax = mode            
            total += distribution.log_likelihood(argmax, i)
    
        return float(total)


class Splitter(object):
    
    def __init__(self):
        self.counter = 0

    def __call__(self, box, max_gumbel):
        bottomcorner, topcorner = box
        self.counter += 1
    
        dim = bottomcorner.shape[0]
        split_dim = np.argmax(topcorner - bottomcorner)
    
        bottom1 = bottomcorner.copy()
        top1 = topcorner.copy()

        bottom2 = bottomcorner.copy()
        top2 = topcorner.copy()
    
        top1[split_dim] = max_gumbel[split_dim]
        bottom2[split_dim] = max_gumbel[split_dim]
        return (bottom1, top1), (bottom2, top2)


def generate_data(N):
    wstar = 2
    x = np.zeros(N)
    y = np.zeros(N)
    x[:N/2] = np.random.randn(N/2)
    y[:N/2] = x[:N/2]*wstar + 0.1*np.random.randn(N/2)
    x[N/2:] = x[:N/2]
    y[:N/2] = -y[:N/2]
    return x, y

def main():
    import time
    sigma = 2 * np.ones(1)

    # bounder
    bounder = Bounder()

    # splitter
    splitter = Splitter()    
    
    # proposal
    proposal = IsotropicGaussian(1, sigma)

    # target
    x, y = generate_data(1000)
    target = CauchyRegression(x, y, sigma)

    # ======== RUN THIS ISH ===============
    
    M = 1
    N = 1000
    print("\nM samples per run:{0}, N runs:{1}".format(M, N))

    start = time.time()
    for i in range(N):
        if i % 100 == 0:
            print i
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

    samples = np.empty((N*M)).squeeze()
    start = time.time()
    for i in range(N):
        if i % 100 == 0:
            print i
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

    samples = samples.ravel()
    plt.hist(samples, bins=25, normed=True)
    p = np.vectorize(lambda x: np.exp(target.log_density(x)))
    x = np.linspace(-5, 5, 1000)
    y = p(x)
    plt.plot(x,y, "r", linewidth=4)
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
