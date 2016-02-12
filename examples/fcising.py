"""Draw a sample from a fully connected Ising model. 
This implementation is research code and could be
made dramatically more efficient."""

import astar
import osstar
import numpy as np
import pulp

class CliqueIsingModel(object):
    
    def __init__(self, w, f):
        self.w = w
        self.n = np.shape(w)[0]
        assert np.size(f) == self.n
        self.f = f
        
    def dimensions(self):
        return self.n
        
    def negative_energy(self, sample):
        pot = 0
        for i in range(self.n):
            pot += self.f[i] if sample[i] == 1 else -self.f[i]
            for j in range(i+1, self.n):
                pot += self.w[i,j] if sample[i] == sample[j] else -self.w[i,j]
        return pot
    
    def make_problem(self, subset):
        assignments, values = subset
        
        prob = pulp.LpProblem("ising", pulp.LpMinimize)
        spin_vars = []
        edge_vars = {}
        
        for i in range(self.n):
            if assignments[i] is None:
                spin_vars.append(pulp.LpVariable("spin" + str(i), 0, 1))
            else:
                spin_vars.append(assignments[i])
            for j in range(i+1, self.n):
                if assignments[i] is not None and assignments[j] is not None:
                    edge_vars[(i,j,0,0)] = 0
                    edge_vars[(i,j,0,1)] = 0
                    edge_vars[(i,j,1,0)] = 0
                    edge_vars[(i,j,1,1)] = 0
                    edge_vars[(i,j,assignments[i],assignments[j])] = 1
                else:
                    edge_vars[(i,j,0,0)] = pulp.LpVariable("edge" + str((i,j,0,0)), 0, 1)
                    edge_vars[(i,j,0,1)] = pulp.LpVariable("edge" + str((i,j,0,1)), 0, 1)
                    edge_vars[(i,j,1,0)] = pulp.LpVariable("edge" + str((i,j,1,0)), 0, 1)
                    edge_vars[(i,j,1,1)] = pulp.LpVariable("edge" + str((i,j,1,1)), 0, 1)
                    
                            
        # objective, energy because we are minimizing
        obj = pulp.lpSum(-self.f[i]*spin_vars[i] + self.f[i]*(1-spin_vars[i]) for i in range(self.n)) 
        obj = obj + pulp.lpSum((-self.w[i,j]*(ci*cj + (1-ci)*(1-cj)) + self.w[i,j]*((1-ci)*cj + ci*(1-cj)))*edge_vars[(i,j,ci,cj)] for i in range(self.n) for j in range(i+1, self.n) for ci in [0, 1] for cj in [0, 1])
        prob += obj
        
        # constraints
        for i in range(self.n):
            for j in range(i+1, self.n):
                if assignments[i] is None or assignments[j] is None:
                    prob += pulp.lpSum(edge_vars[(i,j,0,cj)] for cj in [0, 1]) <= (1 - spin_vars[i])
                    prob += pulp.lpSum(edge_vars[(i,j,ci,0)] for ci in [0, 1]) <= (1 - spin_vars[j])
                    prob += pulp.lpSum(edge_vars[(i,j,1,cj)] for cj in [0, 1]) <= spin_vars[i]
                    prob += pulp.lpSum(edge_vars[(i,j,ci,1)] for ci in [0, 1]) <= spin_vars[j]
        
        return (prob, spin_vars, edge_vars)

    def state_space(self):
        return ([None]*self.n, [0.0]*self.n)
        
class Uniform(object):
            
    def __init__(self):
        self.counter = 0
        
    def log_volume(self, subset):
        assignments, values = subset
        n = sum(i is None for i in assignments)
        return n*np.log(2)

    def negative_energy(self, sample):
        return 0
        
    def sample(self, subset):
        self.counter += 1
        assignments, values = subset
        sample = []
        for i in assignments:
            if i is None:
                sample.append(1 if np.random.rand() > 0.5 else 0)
            else:
                sample.append(i)
        return sample


class Bounder(object):
    
    def __init__(self):
        self.counter = 0

    def __call__(self, distribution, proposal, subset):
        self.counter += 1
        assignments, value = subset
        
        if all(i is not None for i in assignments):
            return distribution.negative_energy(assignments)
        else:
            prob, spin_vars, edge_vars = distribution.make_problem(subset)
            prob.solve(pulp.GLPK(msg = 0))
            for i in range(len(value)):
                if isinstance(spin_vars[i], int):
                    value[i] = spin_vars[i]
                else:
                    value[i] = spin_vars[i].varValue
            return -pulp.value(prob.objective)


class Splitter(object):
    
    def __init__(self):
        self.counter = 0    
        
    def __call__(self, subset, sample):
        self.counter += 1
        assignments, values = subset
    
        closesti = 0
        closest = values[0]
        while assignments[closesti] is not None:
            closesti += 1
            closest = values[closesti]
            
        for i in range(len(values)):
            if assignments[i] is None and abs(0.5 - values[i]) < abs(0.5 - closest):
                closest = values[i]
                closesti = i
    
        left = assignments[:]
        left[closesti] = 0
        assignments[closesti] = 1
    
        return [(left, [0.0]*len(assignments)), (assignments, [0.0]*len(assignments))]
    
        


def main():

    import time
    # bounder
    bounder = Bounder()

    # splitter
    splitter = Splitter()
    
    # proposal
    proposal = Uniform()

    # target
    n = 10
    w = np.triu(np.random.rand(n,n)*0.2, 1)
    f = np.random.rand(n)*2 - 1    
    target = CliqueIsingModel(w, f)
    
            
    # ======== RUN ===============
    
    M = 1
    N = 10
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

    samples = []
    start = time.time()
    for i in range(N):
        if i % 100 == 0:
            print i
        stream = astar.astar_sampling_iterator(target, proposal, bounder, splitter)
        for j in range(M):
            X, G = stream.next()
            samples.append(X)
    end = time.time()


    print("\nA*")
    print("splits/run: {0}".format(splitter.counter/float(N)))
    print("likelihoods/sample: {0}".format(proposal.counter/float(M*N)))
    print("bounds/sample: {0}".format(bounder.counter/float(M*N)))
    print("time taken: {0}s".format(end-start))

if __name__ == '__main__':
    main()