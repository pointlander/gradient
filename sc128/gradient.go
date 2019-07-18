// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sc128

import (
	"math/cmplx"
)

type (
	// V is a value
	V struct {
		X complex128 // the value
		D complex128 // the derivative
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

var (
	sin = cmplx.Sin
	cos = cmplx.Cos
	exp = cmplx.Exp
	log = cmplx.Log
)

// Panic marks a place we should never get to
func Panic(a *V) {
	panic("should not be here")
}

// Meta returns a meta for the value
func (a *V) Meta() Meta {
	return func(k Continuation) Continuation {
		k(a)
		return Panic
	}
}

// Context is a function context
type Context struct {
	InferenceOnly bool
}

// Add adds two numbers
func (context *Context) Add(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X + b.X, 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D
		b.D += c.D
	}
}

// Sub subtracts two numbers
func (context *Context) Sub(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X - b.X, 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D
		b.D -= c.D
	}
}

// Mul multiplies two numbers
func (context *Context) Mul(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X * b.X, 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += b.X * c.D
		b.D += a.X * c.D
	}
}

// Div divides two numbers
func (context *Context) Div(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{a.X / b.X, 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D / b.X
		b.D -= (c.D * a.X) / (b.X * b.X)
	}
}

// Sin the sine of a number
func (context *Context) Sin(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{sin(a.X), 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D * cos(a.X)
	}
}

// Cos the cosine of a number
func (context *Context) Cos(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{cos(a.X), 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D -= c.D * sin(a.X)
	}
}

// Exp the base e exponential
func (context *Context) Exp(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{exp(a.X), 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D * c.X
	}
}

// Log the natural logarithm
func (context *Context) Log(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := V{log(a.X), 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D / a.X
	}
}

// Sigmoid the sigmoid of a number
func (context *Context) Sigmoid(a *V) func(k Continuation) {
	return func(k Continuation) {
		i := exp(a.X)
		c := V{i / (i + 1), 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
		a.D += c.D * c.X * (1 - c.X)
	}
}

// TanH the hyperbolic tangent of a number
func (context *Context) TanH(a *V) func(k Continuation) {
	return func(k Continuation) {
		i, j := exp(a.X), exp(-a.X)
		c := V{(i - j) / (i + j), 0}
		k(&c)
		if context.InferenceOnly {
			return
		}
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
	// Static is the static context
	Static Context
	// Add adds two numbers
	Add = B(Static.Add)
	// Sub subtracts two numbers
	Sub = B(Static.Sub)
	// Mul multiplies two numbers
	Mul = B(Static.Mul)
	// Div divides two numbers
	Div = B(Static.Div)
	// Sin the sine of a number
	Sin = U(Static.Sin)
	// Cos the cosine of a number
	Cos = U(Static.Cos)
	// Exp the base e exponential
	Exp = U(Static.Exp)
	// Log the natural logarithm
	Log = U(Static.Log)
	// Sigmoid the sigmoid of a number
	Sigmoid = U(Static.Sigmoid)
	//TanH the hyperbolic tangent of a number
	TanH = U(Static.TanH)
)

// Gradient computes the gradient
func Gradient(a Meta) (cost V) {
	a(func(a *V) {
		cost = *a
		a.D = 1
	})
	return
}
