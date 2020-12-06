// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package {{.Package}}

import (
	//"math"
  "math/big"

  "github.com/ALTree/bigfloat"
)

type (
	// V is a value
	V struct {
		X big.Float // the value
		D big.Float // the derivative
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
  Precision uint
}

// Add adds two numbers
func (context *Context) Add(k Continuation, a, b *V) bool {
  c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X.Add(&a.X, &b.X)
	if k(&c) {
		return true
	}
	a.D.Add(&a.D, &c.D)
	b.D.Add(&b.D, &c.D)
	return false
}

// Sub subtracts two numbers
func (context *Context) Sub(k Continuation, a, b *V) bool {
  c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X.Sub(&a.X, &b.X)
	if k(&c) {
		return true
	}
	a.D.Sub(&a.D, &c.D)
	b.D.Sub(&b.D, &c.D)
	return false
}

// Mul multiplies two numbers
func (context *Context) Mul(k Continuation, a, b *V) bool {
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X.Mul(&a.X, &b.X)
	if k(&c) {
		return true
	}
  d := big.Float{}
  d.SetPrec(context.Precision)
  d.Mul(&b.X, &c.D)
	a.D.Add(&a.D, &d)
  c.D.Mul(&a.X, &c.D)
	b.D.Add(&b.D, &c.D)
	return false
}

// Div divides two numbers
func (context *Context) Div(k Continuation, a, b *V) bool {
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X.Quo(&a.X, &b.X)
	if k(&c) {
		return true
	}
	a.D.Add(&a.D, c.D.Quo(&c.D, &b.X))
  c.D.Mul(&c.D, &a.X)
  d := big.Float{}
  d.SetPrec(context.Precision)
  d.Mul(&b.X, &b.X)
  c.D.Quo(&c.D, &d)
  b.D.Sub(&b.D, &c.D)
	return false
}

// Sin the sine of a number
func (context *Context) Sin(k Continuation, a *V) bool {
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X = *bigfloat.Sin(&a.X)
	if k(&c) {
		return true
	}
  d := bigfloat.Cos(&a.X)
  d.Mul(&c.D, d)
  a.D.Add(&a.D, d)
	return false
}

// Cos the cosine of a number
func (context *Context) Cos(k Continuation, a *V) bool {
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X = *bigfloat.Cos(&a.X)
	if k(&c) {
		return true
	}
  d := bigfloat.Sin(&a.X)
  d.Mul(&c.D, d)
  a.D.Sub(&a.D, d)
	return false
}

// Exp the base e exponential
func (context *Context) Exp(k Continuation, a *V) bool {
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X = *bigfloat.Exp(&a.X)
	if k(&c) {
		return true
	}
  c.D.Mul(&c.D, &c.X)
  a.D.Add(&a.D, &c.D)
	return false
}

// Log the natural logarithm
func (context *Context) Log(k Continuation, a *V) bool {
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X = *bigfloat.Log(&a.X)
	if k(&c) {
		return true
	}
  c.D.Quo(&c.D, &a.X)
  a.D.Add(&a.D, &c.D)
	return false
}

// Sigmoid the sigmoid of a number
func (context *Context) Sigmoid(k Continuation, a *V) bool {
  i := bigfloat.Exp(&a.X)
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X.Set(i)
  c.X.Add(&c.X, big.NewFloat(1).SetPrec(context.Precision))
  c.X.Quo(i, &c.X)
	if k(&c) {
		return true
	}
  d := big.NewFloat(1).SetPrec(context.Precision)
  d.Sub(d, &c.X)
  d.Mul(d, &c.X)
  d.Mul(d, &c.D)
  a.D.Add(&a.D, d)
	return false
}

// TanH the hyperbolic tangent of a number
func (context *Context) TanH(k Continuation, a *V) bool {
  aa := big.NewFloat(0).SetPrec(context.Precision)
  aa.Set(&a.X)
  aa.Mul(aa, big.NewFloat(-1).SetPrec(context.Precision))
  i, j := bigfloat.Exp(&a.X), bigfloat.Exp(aa)
  x, y := big.NewFloat(0).SetPrec(context.Precision), big.NewFloat(0).SetPrec(context.Precision)
  x.Sub(i, j)
  y.Add(i, j)
	c := V{}
  c.X.SetPrec(context.Precision)
  c.D.SetPrec(context.Precision)
  c.X.Quo(x, y)
	if k(&c) {
		return true
	}
  z := big.NewFloat(0).SetPrec(context.Precision)
  z.Mul(&c.X, &c.X)
  z.Sub(big.NewFloat(1).SetPrec(context.Precision), z)
  z.Mul(z, &c.D)
  a.D.Add(&a.D, z)
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
	Static = Context{
    Precision: 64,
  }
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
		a.D.SetFloat64(1)
		return false
	})
	return
}
