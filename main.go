// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradient

import (
	"math"
)

type (
	// V is a value
	V struct {
		X float64 // the value
		D float64 // the derivative
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

// Add adds two numbers
func Add(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X + b.X, 0}
		k(&c)
		a.D += c.D
		b.D += c.D
	}
}

// Sub subtracts two numbers
func Sub(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X - b.X, 0}
		k(&c)
		a.D += c.D
		b.D -= c.D
	}
}

// Mul multiplies two numbers
func Mul(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X * b.X, 0}
		k(&c)
		a.D += b.X * c.D
		b.D += a.X * c.D
	}
}

// Div divides two numbers
func Div(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X / b.X, 0}
		k(&c)
		a.D += c.D / b.X
		b.D -= (c.D * a.X) / (b.X * b.X)
	}
}

// Sin the sine of a number
func Sin(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{math.Sin(a.X), 0}
		k(&c)
		a.D += c.D * math.Cos(a.X)
	}
}

// Cos the cosine of a number
func Cos(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{math.Cos(a.X), 0}
		k(&c)
		a.D -= c.D * math.Sin(a.X)
	}
}

// Exp the base e exponential
func Exp(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{math.Exp(a.X), 0}
		k(&c)
		a.D += c.D * c.X
	}
}

// Log the natural logarithm
func Log(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{math.Log(a.X), 0}
		k(&c)
		a.D += c.D / c.X
	}
}

// Sigmoid the sigmoid of a number
func Sigmoid(a *V) func(k Continuation) {
	return func(k Continuation) {
		i := math.Exp(a.X)
		c := V{i / (i + 1), 0}
		k(&c)
		a.D += c.D * c.X * (1 - c.X)
	}
}

// TanH the hyperbolic tangent of a number
func TanH(a *V) func(k Continuation) {
	return func(k Continuation) {
		i, j := math.Exp(a.X), math.Exp(-a.X)
		c := V{(i - j) / (i + j), 0}
		k(&c)
		a.D += (1 - c.X*c.X) * c.D
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
	// DivOp divides two numbers
	DivOp = B(Div)
	// SinOp the sine of a number
	SinOp = U(Sin)
	// CosOp the cosine of a number
	CosOp = U(Cos)
	// ExpOp the base e exponential
	ExpOp = U(Exp)
	// LogOp the natural logarithm
	LogOp = U(Log)
	// SigmoidOp the sigmoid of a number
	SigmoidOp = U(Sigmoid)
	//TanHOp the hyperbolic tangent of a number
	TanHOp = U(TanH)
)

// Gradient computes the gradient
func Gradient(a Meta) Continuation {
	return a(func(a *V) {
		a.D = 1
	})
}
