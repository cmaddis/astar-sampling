"""
Rejection and OS* Sampling

OS* Sampling is a new generic sampling algorithm that is an adaptive rejection
algorithm. It is very closely related to A* Sampling as it requires the same
information in order to draw samples.

Adapted from:
    Marc Dymetman, Guillaume Bouchard, and Simon Carter. The OS* Algorithm: a Joint Approach to
    Exact Optimization and Sampling. arXiv preprint arXiv:1207.0742, 2012.
"""

import numpy as np

def rejection_sampling_iterator(distribution, proposal, bounder):
    """An iterator to draw samples from distribution using Rejection Sampling.
    
    Returns: iterator with iterator.next() -> (sample, gumbel)
    sample -- a sample from distribution    
    gumbel -- the next Gumbel from the Gumbel process for distribution
    
    Arguments: distribution, proposal, bounder, splitter
    distribution -- the distribution to sample from
                    This object must have 
                        (1) a distribution.negative_energy(sample) method that
                        returns the negative energy (float) of a sample 
                        (D-dim ndarray).
                        (1) a distribution.state_space() method that returns
                        the state space set (objects of type 'set' will be used
                        by the proposal, bounder, and splitter).
    proposal -- ths distribution to propose from
                This object must have
                    (1) a proposal.log_volume(subset) method that takes the 
                    state space or any subset produced by the splitter function 
                    and returns the log-volume of that subset under proposal,
                    (2) a proposal.negative_energy(sample) method that returns 
                    the negative energy (float) of a sample (D-dim ndarray).
                    (3) a proposal.sample(subset) function that returns 
                    a sample (D-dim ndarray) from the distribution of 
                    the proposal restricted to subset.
    bounder -- the bounding function
               The bounder(distribution, proposal, subset) function must
               return the bound on
               o(sample) = distribution.negative_energy(sample) - 
                           proposal.negative_energy(sample)
               for all sample in subset.
    """

    o = lambda x : distribution.negative_energy(x) - proposal.negative_energy(x)
    state_space = distribution.state_space()
    bound = bounder(distribution, proposal, state_space)
    log_volume = proposal.log_volume(state_space)
    E = 0
    
    while True:                            
        X = proposal.sample(state_space)
        E += np.random.exponential(np.exp(-log_volume))
        if np.random.exponential() > bound - o(X):
            # E is the sum of a variable number of exponentials
            # under a Poisson Process interpretation of Gumbel Processes
            # one can see OS* as performing an adaptive thinning algorithm.
            # This motivates returning -log(E)
            yield X, -np.log(E)
                    
                    
def osstar_sampling_iterator(distribution, proposal, bounder, splitter):
    """An iterator to draw samples from distribution using OS* Sampling.
    
    Returns: iterator with iterator.next() -> (sample, gumbel)
    sample -- a sample from distribution    
    gumbel -- the next Gumbel from the Gumbel process for distribution
    
    Arguments: distribution, proposal, bounder, splitter
    distribution -- the distribution to sample from
                    This object must have 
                        (1) a distribution.dimensions() method that returns
                        the dimensions D of the distribution, and
                        (2) a distribution.negative_energy(sample) method that
                        returns the negative energy (float) of a sample 
                        (D-dim ndarray).
                        (3) a distribution.state_space() method that returns
                        the state space set (objects of type 'set' will be used
                        by the proposal, bounder, and splitter).
    proposal -- ths distribution to propose from
                This object must have
                    (1) a proposal.log_volume(subset) method that takes the 
                    state space or any subset produced by the splitter function 
                    and returns the log-volume of that subset under proposal,
                    (2) a proposal.negative_energy(sample) method that returns 
                    the negative energy (float) of a sample (D-dim ndarray).
                    (3) a proposal.sample(subset) function that returns 
                    a sample (D-dim ndarray) from the distribution of 
                    the proposal restricted to subset.
    bounder -- the bounding function
               The bounder(distribution, proposal, subset) function must
               return the bound on
               o(sample) = distribution.negative_energy(sample) - 
                           proposal.negative_energy(sample)
               for all sample in subset.
    splitter -- the splitting function
                The splitter(subset, sample) returns a list of sets which are
                a partition of subset.    
    """

    o = lambda x : distribution.negative_energy(x) - proposal.negative_energy(x)

    state_space = distribution.state_space()
    bound = bounder(distribution, proposal, state_space)
    subsets_and_bounds = [(state_space, bound)]
    
    E = 0
    
    while True:                            
        subset_energies = np.array([proposal.log_volume(subset) + B for subset, B in subsets_and_bounds])
        i = softmax(subset_energies)
        subset, bound = subsets_and_bounds[i]
        X = proposal.sample(subset)
        E += np.random.exponential(np.exp(-logsumexp(subset_energies)))

        if np.random.exponential() > bound - o(X):
            # E is the sum of a variable number of exponentials
            # under a Poisson Process interpretation of Gumbel Processes
            # one can see OS* as performing an adaptive thinning algorithm.
            # This motivates returning -log(E)
            yield X, -np.log(E)
        else:
            children_subsets = splitter(subset, X)
            for child, child_subset in enumerate(children_subsets):
                subset_and_bound = (child_subset, bounder(distribution, proposal, child_subset))
                if child > 0:
                    subsets_and_bounds.append(subset_and_bound)
                else:
                    subsets_and_bounds[i] = subset_and_bound
        
    
def softmax(x):
    e = np.exp(x - np.max(x))
    Z = e.sum()
    U = np.random.rand()*Z
    return (U > np.cumsum(e)).sum()

def logsumexp(x):
    m = np.max(x)
    return np.log(np.exp(x - m).sum()) + m