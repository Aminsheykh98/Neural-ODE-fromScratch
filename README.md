# Neural ODE from Scratch in Jax

Confession: Before you move on, I have to admit that my own implementation of Dopri5 did not work... Before implementing it myself, I couldn't realize that "the use of mini-batches is less straightforward than for standard neural networks" meant in the paper. The problem is that adaptive solvers like Dopri5 have an if, else statement (if your local error is higher than the specified tolerance, the algorithm should choose another step size that is lower than the previous one). But why was it so hard to implement Dopri5? Well, when you batch your data, each data point in a mini-batch would need a different step size, so the problem with that is how to choose a step size for each different data point in a mini-batch while doing it in parallel. And I still don't have the knowledge to do this. But as far as I'm concerned, the implementation of Jax somehow distributes data in parallel and solves for each data, and before moving on to the next steps of the network, it will wait till all the data is complete (some data might have an easier vector field and can travel very fast from the initial time to the end time, but others may take a lot of steps).

# Motivation
Actually I am a fan of implementing everything from scratch atleast once. As I have noted I did not understand the difficulty of implementing mini-batches before this. I could not find any other "from scratch" implementation on the net so i tried to implement my own version of it and to share.

Final Note: I should note that this implementation is not optimized by any means, and it is just for learning whats behind the curtain.

If you find this helpful please leave a star ;)
