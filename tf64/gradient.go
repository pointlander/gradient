// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tf64

import (
	"math"
)

type (
	// V is a tensor value
	V struct {
		X []float64 // the tensor
		D []float64 // the derivative
		S []int     // the shape
	}
	// Continuation is a continuation
	Continuation func(a *V)
	// Meta is a function that takes a continuation and return a continuation
	Meta func(k Continuation) Continuation
	// Unary is a unary function
	Unary func(a *V) func(k Continuation)
	// Binary is a binary function
	Binary func(a, b *V) func(k Continuation)
)

// Panic marks a place we should never get to
func Panic(a *V) {
	panic("should not be here")
}

// Value returns a meta for the value
func (a *V) Value() Meta {
	return func(k Continuation) Continuation {
		k(a)
		return Panic
	}
}

// Add adds two tensors
func Add(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		size := len(a.X)
		c := V{
			X: make([]float64, 0, size),
			D: make([]float64, size),
			S: []int{a.S[0], a.S[1]},
		}
		for i, j := range a.X {
			c.X = append(c.X, j+b.X[i])
		}
		k(&c)
		for i, j := range c.D {
			a.D[i] += j
			b.D[i] += j
		}
	}
}

// Sub subtracts two tensors
func Sub(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		size := len(a.X)
		c := V{
			X: make([]float64, 0, size),
			D: make([]float64, size),
			S: []int{a.S[0], a.S[1]},
		}
		for i, j := range a.X {
			c.X = append(c.X, j-b.X[i])
		}
		k(&c)
		for i, j := range c.D {
			a.D[i] += j
			b.D[i] -= j
		}
	}
}

// Mul multiplies two tensors
func Mul(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		width := a.S[0]
		if width != b.S[0] {
			panic("first dimension is not the same")
		}
		c := V{
			S: []int{a.S[1], b.S[1]},
		}
		sizeA, sizeC := len(a.X), c.S[0]*c.S[1]
		c.X, c.D = make([]float64, 0, sizeC), make([]float64, sizeC)
		for i := 0; i < sizeC; i += c.S[0] {
			for j := 0; j < sizeA; j += width {
				sum := 0.0
				for k := 0; k < width; k++ {
					sum += a.X[k+j] * b.X[k+i]
				}
				c.X = append(c.X, sum)
			}
		}
		k(&c)
		index := 0
		for i := 0; i < sizeC; i += c.S[0] {
			for j := 0; j < sizeA; j += width {
				for k := 0; k < width; k++ {
					a.D[k+j] += b.X[k+i] * c.D[index]
					b.D[k+i] += a.X[k+j] * c.D[index]
				}
				index++
			}
		}
	}
}

// Sigmoid computes the sigmoid of a vector
func Sigmoid(a *V) func(k Continuation) {
	return func(k Continuation) {
		size := len(a.X)
		c := V{
			X: make([]float64, 0, size),
			D: make([]float64, size),
			S: []int{a.S[0], a.S[1]},
		}
		for _, j := range a.X {
			e := math.Exp(j)
			c.X = append(c.X, e/(e+1))
		}
		k(&c)
		for i, j := range c.D {
			a.D[i] += j * c.X[i] * (1 - c.X[i])
		}
	}
}

// Sum sums a vector
func Sum(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{
			X: make([]float64, 1),
			D: make([]float64, 1),
			S: []int{1, 1},
		}
		for _, j := range a.X {
			c.X[0] += j
		}
		k(&c)
		for i := range a.D {
			a.D[i] += c.D[0]
		}
	}
}

// B converts a binary function into an operator
func B(op Binary) func(a, b Meta) Meta {
	return func(a, b Meta) Meta {
		return func(k Continuation) Continuation {
			return a(func(a *V) {
				b(func(b *V) {
					op(a, b)(k)
				})
			})
		}
	}
}

// U converts a unary function into an operator
func U(op Unary) func(a Meta) Meta {
	return func(a Meta) Meta {
		return func(k Continuation) Continuation {
			return a(func(b *V) {
				op(b)(k)
			})
		}
	}
}

var (
	// AddOp adds two numbers
	AddOp = B(Add)
	// SubOp subtracts two numbers
	SubOp = B(Sub)
	// MulOp multiplies two numbers
	MulOp = B(Mul)
	// SigmoidOp the sigmoid of a number
	SigmoidOp = U(Sigmoid)
	// SumOp sums a vector
	SumOp = U(Sum)
)

// Gradient computes the gradient
func Gradient(a Meta) (cost V) {
	a(func(a *V) {
		cost = *a
		a.D[0] = 1
	})
	return
}
