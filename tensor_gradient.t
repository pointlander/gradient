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
	// V is a tensor value
	V struct {
		X []{{.Type}} // the tensor
		D []{{.Type}} // the derivative
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

// NewV create a new tensor value
func NewV(s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	size := s[0] * s[1]
	return V{
		X: make([]{{.Type}}, 0, size),
		D: make([]{{.Type}}, size),
		S: s,
	}
}

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

// Zero zeros the partial derivatives
func (a *V) Zero() {
	for i := range a.D {
		a.D[i] = 0
	}
}

// Set sets the values and zeros the partial derivatives
func (a *V) Set(values []{{.Type}}) {
	for i, value := range values {
		if i >= len(a.X) {
			a.X = append(a.X, value)
			continue
		}
		a.X[i] = value
	}
	a.Zero()
}

// Context is a function context
type Context struct {
	InferenceOnly bool
}

// Add adds two tensors
func (context *Context) Add(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c := NewV(a.S...)
		for i, j := range a.X {
			c.X = append(c.X, j+b.X[i])
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j
			b.D[i] += j
		}
	}
}

// Sub subtracts two tensors
func (context *Context) Sub(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c := NewV(a.S...)
		for i, j := range a.X {
			c.X = append(c.X, j-b.X[i])
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j
			b.D[i] -= j
		}
	}
}

// Mul multiplies two tensors
func (context *Context) Mul(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		width := a.S[0]
		if width != b.S[0] {
			panic("first dimension is not the same")
		}
		c := NewV(a.S[1], b.S[1])
		sizeA, sizeB := len(a.X), len(b.X)
		for i := 0; i < sizeB; i += width {
			bv := b.X[i : i+width]
			for j := 0; j < sizeA; j += width {
				av, sum := a.X[j:j+width], {{.Type}}(0.0)
				for k, bx := range bv {
					sum += av[k] * bx
				}
				c.X = append(c.X, sum)
			}
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		index := 0
		for i := 0; i < sizeB; i += width {
			bv, bd := b.X[i:i+width], b.D[i:i+width]
			for j := 0; j < sizeA; j += width {
				av, ad := a.X[j:j+width], a.D[j:j+width]
				for k, bx := range bv {
					ad[k] += bx * c.D[index]
					bd[k] += av[k] * c.D[index]
				}
				index++
			}
		}
	}
}

// Hadamard computes the hadamard product of two tensors
func (context *Context) Hadamard(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c := NewV(a.S...)
		for i, j := range a.X {
			c.X = append(c.X, j*b.X[i])
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j * b.X[i]
			b.D[i] += j * a.X[i]
		}
	}
}

// Sin the sine of a number
func (context *Context) Sin(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S...)
		for _, j := range a.X {
			c.X = append(c.X, sin(j))
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j * cos(a.X[i])
		}
	}
}

// Cos the cosine of a tensor
func (context *Context) Cos(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S...)
		for _, j := range a.X {
			c.X = append(c.X, cos(j))
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] -= j * sin(a.X[i])
		}
	}
}

// Exp the base e exponential of a tensor
func (context *Context) Exp(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S...)
		for _, j := range a.X {
			c.X = append(c.X, exp(j))
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j * c.X[i]
		}
	}
}

// Log the natural logarithm of a tensor
func (context *Context) Log(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S...)
		for _, j := range a.X {
			c.X = append(c.X, log(j))
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j / a.X[i]
		}
	}
}

// Sigmoid computes the sigmoid of a vector
func (context *Context) Sigmoid(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S...)
		for _, j := range a.X {
			e := exp(j)
			c.X = append(c.X, e/(e+1))
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			cx := c.X[i]
			a.D[i] += j * cx * (1 - cx)
		}
	}
}

// TanH the hyperbolic tangent of a tensor
func (context *Context) TanH(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S...)
		for _, j := range a.X {
			e1, e2 := exp(j), exp(-j)
			c.X = append(c.X, (e1-e2)/(e1+e2))
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			cx := c.X[i]
			a.D[i] += j * (1 - cx*cx)
		}
	}
}

// Softmax is the softmax function
func (context *Context) Softmax(a *V) func(k Continuation) {
	return func(k Continuation) {
		c, sum := NewV(a.S...), {{.Type}}(0.0)
		for _, j := range a.X {
			e := exp(j)
			sum += e
			c.X = append(c.X, e)
		}
		for i, j := range c.X {
			c.X[i] = j / sum
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			cx := c.X[i]
			a.D[i] += j * (cx - cx*cx)
		}
	}
}

// Sum sums a vector
func (context *Context) Sum(a *V) func(k Continuation) {
	return func(k Continuation) {
		c, sum := NewV(1), {{.Type}}(0.0)
		for _, j := range a.X {
			sum += j
		}
		c.X = append(c.X, sum)
		k(&c)
		if context.InferenceOnly {
			return
		}
		d := c.D[0]
		for i := range a.D {
			a.D[i] += d
		}
	}
}

// Quadratic computes the quadratic cost of two tensors
func (context *Context) Quadratic(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c, sum := NewV(1), {{.Type}}(0.0)
		for i, j := range a.X {
			p := (j - b.X[i])
			sum += p * p
		}
		c.X = append(c.X, .5 * sum)
		k(&c)
		if context.InferenceOnly {
			return
		}
		d := c.D[0]
		for i, j := range a.X {
			a.D[i] += (j - b.X[i]) * d
			b.D[i] += (b.X[i] - j) * d
		}
	}
}

// CrossEntropy computes the cross entropy cost of two tensors
func (context *Context) CrossEntropy(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c, sum := NewV(1), {{.Type}}(0.0)
		for i, ax := range a.X {
			bx := b.X[i]
			if bx == 1 {
				sum += log(ax + .001)
			} else {
				sum += log(1 - ax + .001)
			}
		}
		c.X = append(c.X, -sum)
		k(&c)
		if context.InferenceOnly {
			return
		}
		d := c.D[0]
		for i, ax := range a.X {
			bx := b.X[i]
			if bx == 1 {
				a.D[i] -= d / (ax + .001)
				b.D[i] -= log(ax + .001) * d
			} else {
				a.D[i] += d / (1 - ax + .001)
				b.D[i] -= log(1 - ax + .001) * d
			}
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
	// Static is the static context
	Static Context
	// Add adds two tensors
	Add = B(Static.Add)
	// Sub subtracts two tensors
	Sub = B(Static.Sub)
	// Mul multiplies two tensors
	Mul = B(Static.Mul)
	// Hadamard computes the hadamard product of two tensors
	Hadamard = B(Static.Hadamard)
	// Sin the sin of a tensors
	Sin = U(Static.Sin)
	// Cos the cosine of a tensor
	Cos = U(Static.Cos)
	// Exp the base e exponential of a tensor
	Exp = U(Static.Exp)
	// Log the natural logarithm of a tensor
	Log = U(Static.Log)
	// Sigmoid the sigmoid of a tensors
	Sigmoid = U(Static.Sigmoid)
	// TanH the hyperbolic tangent of a tensor
	TanH = U(Static.TanH)
	// Softmax is the softmax function
	Softmax = U(Static.Softmax)
	// Sum sums a vector
	Sum = U(Static.Sum)
	// Quadratic computes the quadratic cost of two tensors
	Quadratic = B(Static.Quadratic)
	// CrossEntropy computes the cross entropy cost of two tensors
	CrossEntropy = B(Static.CrossEntropy)
)

// Gradient computes the gradient
func Gradient(a Meta) (cost V) {
	a(func(a *V) {
		cost = *a
		a.D[0] = 1
	})
	return
}
