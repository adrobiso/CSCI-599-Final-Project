# CSCI-599-Final-Project: Evolutionary Emergence of Grounded Compositional Language

Multiagent environment code from: [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)  
PyTorch NEAT code from: [PyTorch-NEAT](https://github.com/uber-research/PyTorch-NEAT)

## Introduction
In 2018, Mordatch and Abbeel demonstrated the emergence of a grounded, compositional language in a multi-agent environment without any explicit reward for communication or exposure to human language (Mordatch and Abbeel, 2018). They proposed and developed a multi-agent environment in which language is neccesary for a solution to be reached, without explicitly rewarding language formation. For their agents, they utilized shared-policy gradient methods with distributed execution, with a network structure defined A Priori.

Modern neuro-evolution techniques such as NEAT, and other NEAT-based algorithms, have been shown to arrive at solutions faster than reinforcement learning methods for some problems (Stanley and Miikkulainen, 2002). Additionally, many of these methods evolve the structure of the network simultaniously to the weights starting from a minimal model, often a fully-connected network with no hidden layers.

In this work, the application of NEAT to the environment proposed and developed in Mordatch and Abbeel (2018) is explored to determine if, and how quickly a solution is reached. If successful, the resulting network structures could provide insight into more optimal A Priori network designs for other tasks such as NLP as well.

## Problem Formulation
The environment is the environment used in Mordatch and Abbeel (2018), a partially-observable Markov game with continuous space and discrete time. The setup used is “simple reference”, which consists of five entities, three landmarks and two agents, each having a color and a position. Each entity is initialized with a random position, and each landmark with a unique color. Each agent is initialized with a goal vector comprised of a target agent and the color of a target landmark, and agents are rewarded based on the distance of the target agent from the target landmark after 100 time steps. Each Agent’s color is set to the color of the target landmark in the other agent’s goal vector, or more simply: the landmark the actor would idealy go to.

The observation space for an agent is comprised of the color of the goal vector (as there’s only one other agent to be the target), the position of each landmark relative to the agent’s position, the color of each landmark, and the utterances of the other agent.

At each timestep, agents can perform a physical action and a communication action. Physical actions are the applications of a directional force. There is no collision between entities. Communication actions are utterances of a discrete symbol observed by all other agents. Agents can also choose not to utter a symbol.

<img src="https://raw.githubusercontent.com/adrobiso/CSCI-599-Final-Project/master/images/environment.png" width="400">

Example view of the environment. Landmarks are opaque, while agents are slightly transparent.

## Models and Approaches

Each actor's network is instantiated from a genome model evolved using NEAT (Stanley and Miikkulainen, 2002). Initially, genomes encode minimal networks, with no hidden neurons, but evolve to become more complex over generations. Each genome in the population is fitness-evaluated by averaging over 10 runs, guiding NEAT’s reproduction and speciation schemes for the next generation

As in Mordatch and Abbeel (2018), communication utterances were determined using a Gumbel-SoftMax estimator, with the networks outputting the log-probabilities of each symbol (including no-utterance). Also as in Mordatch and Abbeel (2018), during fitness-evaluation two auxiliary reward functions for goal prediction and vocabulary size penalization were used.

As a replacement for NEAT, HyperNEAT was also investigated as a potential approach, but was quickly found to be very slow both in this work and in Lowell, Grabkovsky, and Birger (2011).

## Results
Promisingly, the networks were qualitatively observed to quickly learn to move to approximately the centroid of all landmarks, a behavior noted in Mordatch and Abbeel (2018). 

<video width="320" height="240" controls>
  <source src="centroid.mp4" type="video/mp4">
</video>

At this point however, progress completely stalls and a grounded compositional language never emerges. The genomes themselves are not completely stagnant though: the number of connections in each slowly decrease without any apparent impact on performance. This is particularly noteworthy as there is no penalty for complexity in NEAT, and the chances of adding and deleting a connection are the same, as well as for activating and deactivating a connection. This suggests that the simpler networks actually tend to perform slightly better than the more complex ones.

<img src="https://raw.githubusercontent.com/adrobiso/CSCI-599-Final-Project/master/images/plots.png">

The networks also appear to begin to learn that smaller vocabularies are better, as the frequency of one symbol far exceed the rest, though it does so by choosing not to utter a symbol far less.

<img src="https://raw.githubusercontent.com/adrobiso/CSCI-599-Final-Project/master/images/nkbarsinitial.png">
<img src="https://raw.githubusercontent.com/adrobiso/CSCI-599-Final-Project/master/images/nkbars.png">

## Discussion and Future Work

There are a couple prominent possible reasons NEAT failed to develop a grounded compositional language. 

First, rather than evolving with increasing complexity as is intended with NEAT, the networks’ complexity actually decreased over generations. This, along with the stagnation in performance beyond the centroid behavior, is indicative that the problem space is too complex for standard NEAT to quickly solve, as progress was only made on the most basic dimensions.

Second, there is no explicit memory bank as in Mordatch and Abbeel (2018). NEAT can evolve recurrent connections, but that option was disabled for this experiment.  For future work, allowing recurrent connections could assist the networks in developing their own memory banks, possibly allowing for better performance.

## References
Kenneth O Stanley and Risto Miikkulainen. Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2):99-127, 2002.

Igor Mordatch and Pieter Abeel. Emergence of Grounded Compositional Language in Multi-Agent Populations. In *Thirty-Second AAAI Congerence on Artificial Intelligence*, 2018.

J. Lowell, S. Grabkovsky and K. Birger, "Comparison of NEAT and HyperNEAT Performance on a Strategic Decision-Making Problem," 2011 Fifth International Conference on Genetic and Evolutionary Computing, Xiamen, pp. 102-105, 2011.
