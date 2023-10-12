# Introduction to Artificial Intelligence

## note 1

rational agent: perform **actions** that yield the **best/optimal** expected **outcome** given the **goals**.

### Search Problem

How to define one?

- State space: all possible states possible in the given world.
- Successor function: take in **state** and **action**, compute the **cost** of performing the action and the **successor state**.
- Start state
- Goal test: judge if the input **state** is a goal state.

Space size: $\prod_{i=1}^{n} x_{i}$, while $x_{i}$ is the numbers of value the i-th object can take.

Space size is needed to estimate the computational runtime of solving certain problem.

### Space State Graph and Search Tree

Graph: nodes and edges(weight)

Space State Graph:

- nodes: space state, **each space state only appear once**
- edges: actions, cost of action as weight.

Search Tree:

a class of graph but **each space state can appear for maybe more than once.**

a branch represents a path or a plan.
