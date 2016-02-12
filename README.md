# A* Sampling

## Overview

A* Sampling is a generic sampling algorithm 
based on the Gumbel-Max trick. It relies on 
a proposal distribution as well as bounds on
the log ratio of densities. This is the same
information used by rejection sampling 
algorithms. If the tightness of bounds 
improves with shrinking region volume, then
A* sampling is asymptotically efficient ---
in the limit, for every proposal consumed 
one sample is produced. This does not mean 
that the method is efficient for high 
dimensions. The runtime to the first sample 
scales exponentially with dimension. The 
Python code includes a generator 
implementation of A* sampling that can be
used to produce any specified number of 
samples. Also it includes an implementation of rejection
sampling and OS*.

## Dependencies
  - Python 2.7
  - Numpy
  - Scipy
  - Matplotlib
  - [Pulp](https://github.com/coin-or/pulp)

## Running examples
For example

    cd astar-sampling
    python examples/sin.py

## Citation
If you use this code for published work 
please cite

Chris J. Maddison, Daniel Tarlow, Tom Minka.
A* Sampling. NIPS 2014.

    @incollection{maddison2014astarsamp,
        title = "{A$^\ast$  Sampling}",
        author = {Maddison, Chris J and Tarlow, Daniel and Minka, Tom},
        booktitle = {Advances in Neural Information Processing Systems 27},
        year = {2014},
    }
