// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package {{.Package}}

import (
{{if eq .Type "float64"}}
	"math"
{{else if eq .Type "float32"}}
  "math"
{{else if eq .Type "complex128"}}
  "math/cmplx"
{{end}}
)

type (
	// V is a value
	V struct {
		X {{.Type}} // the value
		D {{.Type}} // the derivative
	}
	// Continuation is a continuation
	Continuation func(a *V) bool
	// Meta is a function that takes a continuation and return a continuation
	Meta func(k Continuation) Continuation
	// Unary is a unary function
	Unary func(k Continuation, a *V) bool
	// Binary is a binary function
	Binary func(k Continuation, a, b *V) bool
)

{{if eq .Type "float64"}}
var (
	sin = math.Sin
	cos = math.Cos
	exp = math.Exp
	log = math.Log
)
{{else if eq .Type "float32"}}
func sin(a float32) float32 {
	return float32(math.Sin(float64(a)))
}
func cos(a float32) float32 {
	return float32(math.Cos(float64(a)))
}
func exp(a float32) float32 {
	return float32(math.Exp(float64(a)))
}
func log(a float32) float32 {
	return float32(math.Log(float64(a)))
}
{{else if eq .Type "complex128"}}
var (
	sin = cmplx.Sin
	cos = cmplx.Cos
	exp = cmplx.Exp
	log = cmplx.Log
)
{{end}}

// Panic marks a place we should never get to
func Panic(a *V) bool {
	panic("should not be here")
	return false
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

}

// Add adds two numbers
func (context *Context) Add(k Continuation, a, b *V) bool {
	c := V{a.X + b.X, 0}
	if k(&c) {
		return true
	}
	a.D += c.D
	b.D += c.D
	return false
}

// Sub subtracts two numbers
func (context *Context) Sub(k Continuation, a, b *V) bool {
	c := V{a.X - b.X, 0}
	if k(&c) {
		return true
	}
	a.D += c.D
	b.D -= c.D
	return false
}

// Mul multiplies two numbers
func (context *Context) Mul(k Continuation, a, b *V) bool {
	c := V{a.X * b.X, 0}
	if k(&c) {
		return true
	}
	a.D += b.X * c.D
	b.D += a.X * c.D
	return false
}

// Div divides two numbers
func (context *Context) Div(k Continuation, a, b *V) bool {
	c := V{a.X / b.X, 0}
	if k(&c) {
		return true
	}
	a.D += c.D / b.X
	b.D -= (c.D * a.X) / (b.X * b.X)
	return false
}

// Sin the sine of a number
func (context *Context) Sin(k Continuation, a *V) bool {
	c := V{sin(a.X), 0}
	if k(&c) {
		return true
	}
	a.D += c.D * cos(a.X)
	return false
}

// Cos the cosine of a number
func (context *Context) Cos(k Continuation, a *V) bool {
	c := V{cos(a.X), 0}
	if k(&c) {
		return true
	}
	a.D -= c.D * sin(a.X)
	return false
}

// Exp the base e exponential
func (context *Context) Exp(k Continuation, a *V) bool {
	c := V{exp(a.X), 0}
	if k(&c) {
		return true
	}
	a.D += c.D * c.X
	return false
}

// Log the natural logarithm
func (context *Context) Log(k Continuation, a *V) bool {
	c := V{log(a.X), 0}
	if k(&c) {
		return true
	}
	a.D += c.D / a.X
	return false
}

// Sigmoid the sigmoid of a number
func (context *Context) Sigmoid(k Continuation, a *V) bool {
	i := exp(a.X)
	c := V{i / (i + 1), 0}
	if k(&c) {
		return true
	}
	a.D += c.D * c.X * (1 - c.X)
	return false
}

// TanH the hyperbolic tangent of a number
func (context *Context) TanH(k Continuation, a *V) bool {
	i, j := exp(a.X), exp(-a.X)
	c := V{(i - j) / (i + j), 0}
	if k(&c) {
		return true
	}
	a.D += (1 - c.X*c.X) * c.D
	return false
}

// B converts a binary function into an operator
func B(op Binary) func(a, b Meta) Meta {
	return func(a, b Meta) Meta {
		return func(k Continuation) Continuation {
			return a(func(a *V) bool {
				derivatives := false
				b(func(b *V) bool {
					derivatives = op(k, a, b)
					return derivatives
				})
				return derivatives
			})
		}
	}
}

// U converts a unary function into an operator
func U(op Unary) func(a Meta) Meta {
	return func(a Meta) Meta {
		return func(k Continuation) Continuation {
			return a(func(b *V) bool {
				return op(k, b)
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
	a(func(a *V) bool {
		cost = *a
		a.D = 1
		return false
	})
	return
}
