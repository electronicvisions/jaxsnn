# jaxsnn

This library is splitted into two parts. A conventional time-stepping approach and an event-based library, found in the `event` folder.

## Event Library

### Notebook

Multiple notebooks in the `event` part demonstrate simples parts of the functionality.

`leaky_integrate` builds a simple leaky integrator layer, defines a loss and trains it
`ttfs_gradient` derives an analytic solution for neural dynamics and shows a simple gradient descent through the root solver
`ttfs` explores the dynamics of the analytical ttfs solution and inspects the gradient
`neural_net` writes down all the neuron dynamics and trains a simple two layered net on multiple tasks and with two different losses (max over time / ttfs)

### Tasks

Run with 

```bash
python -m jaxsnn.event.tasks.first_spike
```

`gradients` inspects the gradients of different abstraction levels of the library
`first_spike` trains a net with a loss dependant on the first spike time
`max_over_time` trains a net with a loss dependant on the max voltage of a leaky integrate layer

### Current problems

The loss does not s
