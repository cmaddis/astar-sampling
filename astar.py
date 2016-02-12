"""
A* Sampling

A* Sampling is a generic sampling algorithm based on the Gumbel-Max trick.
It relies on a proposal distribution as well as bounds on the log ratio of
densities. This is the same information consumed by rejection sampling algorithms. 
If the tightness of bounds improves with shrinking region volume, then
A* sampling is asymptotically efficient --- in the limit, for every proposal 
consumed one sample is produced. This does not mean that the method is efficient 
for high dimensions. The runtime to the first sample scales exponentially 
with dimension.

astar_sampling_iterator(distribution, proposal, bounder, splitter)
    An iterator which returns an indefinite number of samples from distribution. 
    
See sin.py or softmax.py or clutter.py for examples of use.
"""

__version__ = '1.0.1'

from itertools import imap
import heaps
import numpy as np


def astar_sampling_iterator(distribution, proposal, bounder, splitter):
    """An iterator to draw samples from distribution using A* Sampling.
    
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


    # the difference in energy functions o(x) needs to be boundable
    o = lambda x : distribution.negative_energy(x) - proposal.negative_energy(x)

    # get the state space
    state_space = distribution.state_space()
    bound = bounder(distribution, proposal, state_space)
    log_volume = proposal.log_volume(state_space)
    
    # inititalize the max-heap of upperbounds
    upperbounds = [(gumbel(log_volume) + bound, state_space, bound, log_volume)]
    
    # inititalize the max-heap of lowerbounds
    lowerbounds = []

    while True:
        
        # pick the region with maximum upperbound
        node = heaps.maxheappop(upperbounds)
        upperbound, subset, bound, log_volume = node
        
        # sample within that region
        X = proposal.sample(subset)
            
        # recover the true gumbel value
        G = upperbound - bound

        # compute the lower bound
        heaps.maxheappush(lowerbounds, (G + o(X), X))
        
        # now we try to advance the current Gumbel chain
        G = truncate(gumbel(log_volume), G)
        
        # we want the following condition to hold:
        #       if distribution == proposal, then never split
        # the following split condition ensures that
        
        replacement = (G + bound, subset, bound, log_volume)
        split = heaps.maxheappeek(lowerbounds) < max(replacement, heaps.maxheappeek(upperbounds))
        
        
        if split:
            children_subsets = splitter(subset, X)
            children_log_volumes = np.array(map(proposal.log_volume, children_subsets))

            # we already sampled a Gumbel that one of the children needs 
            # to inherit. This decision must be stochastic. It could also be
            # done more efficiently by sample from the proposal over the parent
            # region and then assigning it to the child whose subset contains 
            # the sample. This would require breaking the convention that
            # we only sample from the proposal when a region is taken off
            # the upperbounds max-heap.
            heir = softmax(children_log_volumes)

            for child, child_subset in enumerate(children_subsets):
                child_bound = bounder(distribution, proposal, child_subset)
                child_log_volume = children_log_volumes[child]
                if child == heir:
                    child_gumbel = G
                else:
                    child_gumbel = truncate(gumbel(child_log_volume), G)
                child_node = (child_gumbel + child_bound, 
                                child_subset, 
                                child_bound,
                                child_log_volume)
                                
                heaps.maxheappush(upperbounds, child_node)
                                                
        else:
            # since we didn't split we can advance the chain
            heaps.maxheappush(upperbounds, replacement)
        
        
        # Recall that every sample on the lowerbound heap whose lowerbound
        # exceeds the max upperbound is a sample. It is true that
        # we may have multiple samples to return, however in this 
        # implementation we only return once for every sample considered. 
        # This might seem counterintuitive but it balances how quickly 
        # we deplete out lowerbound heap with how quickly the upperbound descends.
         
        if heaps.maxheappeek(lowerbounds) >= heaps.maxheappeek(upperbounds):
            G, X = heaps.maxheappop(lowerbounds)
            yield X, G
                    
def gumbel(mu):
    """Return an independent sample from Gumbel(mu)."""
    return -np.log(np.random.exponential()) + mu
    
def truncgumbel(mu, b):
    """Return an independent sample from TruncGumbel(mu, b)."""    
    return -np.log(np.random.exponential() + np.exp(-b + mu)) + mu
    
def softmax(x):
    """Return an independent sample from Softmax(x)."""
    e = np.exp(x - np.max(x))
    Z = e.sum()
    U = np.random.rand()*Z
    return (U > np.cumsum(e)).sum()

def logsumexp(x):
    """Return log(exp(x).sum())."""
    m = np.max(x)
    return np.log(np.exp(x - m).sum()) + m

def truncate(G, b):
    """(G ~ Gumbel(mu), b) -> TruncGumbel(mu, b)"""
    return -np.log(np.exp(-G) + np.exp(-b))
    
def untruncate(G, b):
    """(G ~ TruncGumbel(mu, b), b) -> Gumbel(mu)"""
    return G - np.log(1 - np.exp(G-b))