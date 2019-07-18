[![godoc](https://godoc.org/github.com/pointlander/gradient?status.svg)](https://godoc.org/github.com/pointlander/gradient)

# Reverse Mode Automatic Differentiation with Continuation Passing Style

This project implements [reverse mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) with [continuation passing style](https://en.wikipedia.org/wiki/Continuation-passing_style) (CPS). Tensors and scalars are both supported with number type of float64, float32, or complex128. Gradient was inspired by [Lantern](https://feiwang3311.github.io/Lantern/) as described by this [paper](http://papers.nips.cc/paper/8221-backpropagation-with-callbacks-foundations-for-efficient-and-expressive-differentiable-programming.pdf).
