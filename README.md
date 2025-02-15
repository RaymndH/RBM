# Restricted Boltzmann Machine Codes

Python code to investigate usage of a restricted Boltzmann machine (RBM) on solving Ising Problems of various hardness

An RBM is a type of neural network which has two sets of nodes which are in essence copies of each other. Nodes in group A are connected only to nodes in group B and vice versa. Because each node can be in one of two states, on or off, it is possible to model the energy of an Ising problem by transforming the weights in the problem to corresponding weights and biases in an RBM. Each iteration performs Gibbs Sampling on each neuron and is accepted with some probability, tending towards lower energies. In this way, the most likely state can be found, and lower energy states can be recorded. 

Included are Wishart examples of various hardness. Wishart problems show promise in being able to generate problems of arbitrary hardness with relatively small sizes.

![New Project](https://github.com/user-attachments/assets/4ad4ddf9-e91d-4637-8e03-361d3d1bbc18)

Example of a N=5 Wishart problem as an RBM. Each of the edges connected the nodes corresponds to a value from the matrix. Information flows in two stages, from the hidden layer to the visible layer and back, eventually converging on a state which will usually have a low Ising energy.
