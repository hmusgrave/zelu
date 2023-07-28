# zelu

vectorized elu activation

## Purpose

Elu is fairly simple (e^x-1 for negative inputs, x otherwise), and its gradient is even easier. It tends to skew the data though, transforming a whitened gaussian into something with nonzero mean and nonunit variance. This library offers a simple vectorized elu implementation (and its gradient) and also has the ability to whiten inputs.

## Status

The code works, but I have other things to do today. I'll make a release and document it later.
