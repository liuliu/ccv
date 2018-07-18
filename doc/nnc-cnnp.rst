NNC Common Neural Network Primitives
====================================

Computation graph is a powerful abstraction for scheduling and managing computations. Often times though, this can feel too *raw*. Conceptually, in a computation graph, all tensors are equal. But for neural networks, parameters (weights) and activations are different. Parameters are the configurations, while activations are the temporary states given the input.

Both weights and activations in computation graph are represented as ordinary tensors.

Model
-----

**Model** is the core abstraction in **common neural network primitives (CNNP)** interface. It can represent both a layer or a group of layers. An ordinary neural network layer contains parameters, and applies the parameters to input neurons to generate activations in output neurons. The **model** abstraction goes beyond one input and one output. It can take multiple sets of input neurons, and generate activations on multiple sets of output neurons. The main difference between a **model** and a **command** in a concrete graph is that the **model** contains states (parameters).
