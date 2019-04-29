# CSCI-599-Final-Project: Evolutionary Emergence of Grounded Compositional Language

Multiagent environment code from: https://github.com/openai/multiagent-particle-envs  
PyTorch NEAT code from: https://github.com/uber-research/PyTorch-NEAT

## Introduction
In 2018, Mordatch and Abbeel demonstrated the emergence of a grounded, compositional language in a multi-agent environment without any explicit reward for communication or exposure to human language (Mordatch and Abbeel, 2018). They proposed and developed a multi-agent environment in which language is neccesary for a solution to be reached, without explicitly rewarding language formation. For their agents, they utilized shared-policy gradient methods with distributed execution, with a network structure defined A Priori.

Modern neuro-evolution techniques such as NEAT, and other NEAT-based algorithms, have been shown to arrive at solutions faster than reinforcement learning methods for some problems (Stanley and Miikkulainen, 2002). Additionally, many of these methods evolve the structure of the network simultaniously to the weights starting from a minimal model, often a fully-connected network with no hidden layers.

In this work, the application of NEAT to the environment proposed and developed in Mordatch and Abbeel (2018) is explored to determine if, and how quickly a solution is reached. If successful, the resulting network structures could provide insight into more optimal A Priori network designs for other tasks such as NLP as well.

## Problem Formulation

## Experiments

## References
Kenneth O Stanley and Risto Miikkulainen. Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2):99-127, 2002.
Igor Mordatch and Pieter Abeel. Emergence of Grounded Compositional Language in Multi-Agent Populations. In *Thirty-Second AAAI Congerence on Artificial Intelligence*, 2018.
