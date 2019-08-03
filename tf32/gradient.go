// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tf32

import (
	"math"
)

type (
	// V is a tensor value
	V struct {
		X []float32 // the tensor
		D []float32 // the derivative
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

// NewV create a new tensor value
func NewV(s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	size := s[0] * s[1]
	return V{
		X: make([]float32, 0, size),
		D: make([]float32, size),
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
func (a *V) Set(values []float32) {
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
		width, length := a.S[0], len(b.X)
		if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
			panic("dimensions are not the same")
		}
		c := NewV(a.S...)
		for i, j := range a.X {
			c.X = append(c.X, j+b.X[i%length])
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j
			b.D[i%length] += j
		}
	}
}

// Sub subtracts two tensors
func (context *Context) Sub(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		width, length := a.S[0], len(b.X)
		if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
			panic("dimensions are not the same")
		}
		c := NewV(a.S...)
		for i, j := range a.X {
			c.X = append(c.X, j-b.X[i%length])
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, j := range c.D {
			a.D[i] += j
			b.D[i%length] -= j
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
				av, sum := a.X[j:j+width], float32(0.0)
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

// T the transpose of the matrix
func (context *Context) T(a *V) func(k Continuation) {
	return func(k Continuation) {
		c := NewV(a.S[1], a.S[0])
		for p := 0; p < a.S[0]; p++ {
			for q := 0; q < a.S[1]; q++ {
				c.X = append(c.X, a.X[q*a.S[0]+p])
			}
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		i := 0
		for p := 0; p < a.S[0]; p++ {
			for q := 0; q < a.S[1]; q++ {
				a.D[q*a.S[0]+p] = c.D[i]
				i++
			}
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
		c, size, width := NewV(a.S...), len(a.X), a.S[0]
		for i := 0; i < size; i += width {
			sum := float32(0.0)
			for _, ax := range a.X[i : i+width] {
				e := exp(ax)
				sum += e
				c.X = append(c.X, e)
			}
			for j, cx := range c.X[i : i+width] {
				c.X[i+j] = cx / sum
			}
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		for i, d := range c.D {
			cx := c.X[i]
			a.D[i] += d * (cx - cx*cx)
		}
	}
}

// Sum sums a vector
func (context *Context) Sum(a *V) func(k Continuation) {
	return func(k Continuation) {
		c, sum := NewV(1), float32(0.0)
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
		width := a.S[0]
		if width != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c, size := NewV(a.S[1]), len(a.X)
		for i := 0; i < size; i += width {
			av, bv, sum := a.X[i:i+width], b.X[i:i+width], float32(0.0)
			for j, ax := range av {
				p := (ax - bv[j])
				sum += p * p
			}
			c.X = append(c.X, .5*sum)
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		index := 0
		for i := 0; i < size; i += width {
			av, bv, ad, bd, d := a.X[i:i+width], b.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
			for j, ax := range av {
				ad[j] += (ax - bv[j]) * d
				bd[j] += (bv[j] - ax) * d
			}
			index++
		}
	}
}

// CrossEntropy computes the cross entropy cost of two tensors
func (context *Context) CrossEntropy(a, b *V) func(k Continuation) {
	return func(k Continuation) {
		if len(a.S) != 2 || len(b.S) != 2 {
			panic("tensor needs to have two dimensions")
		}
		width := a.S[0]
		if width != b.S[0] || a.S[1] != b.S[1] {
			panic("dimensions are not the same")
		}
		c, size := NewV(a.S[1]), len(a.X)
		for i := 0; i < size; i += width {
			av, bv, sum := a.X[i:i+width], b.X[i:i+width], float32(0.0)
			for j, ax := range av {
				bx := bv[j]
				if bx == 1 {
					sum += log(ax + .001)
				} else {
					sum += log(1 - ax + .001)
				}
			}
			c.X = append(c.X, -sum)
		}
		k(&c)
		if context.InferenceOnly {
			return
		}
		index := 0
		for i := 0; i < size; i += width {
			av, bv, ad, bd, d := a.X[i:i+width], b.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
			for j, ax := range av {
				bx := bv[j]
				if bx == 1 {
					ad[j] -= d / (ax + .001)
					bd[j] -= log(ax+.001) * d
				} else {
					ad[j] += d / (1 - ax + .001)
					bd[j] -= log(1-ax+.001) * d
				}
			}
			index++
		}
	}
}

// Avg computes the average of the tensor
func (context *Context) Avg(a *V) func(k Continuation) {
	return func(k Continuation) {
		c, sum := NewV(1), float32(0.0)
		for _, j := range a.X {
			sum += j
		}

		total := float32(len(a.X))

		c.X = append(c.X, sum/total)
		k(&c)
		if context.InferenceOnly {
			return
		}
		d := c.D[0] / total
		for i := range a.D {
			a.D[i] += d
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
	// // T the transpose of the matrix
	T = U(Static.T)
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
	// Avg computes the average of the tensor
	Avg = U(Static.Avg)
)

// Gradient computes the gradient
func Gradient(a Meta) (cost V) {
	a(func(a *V) {
		cost = *a
		a.D[0] = 1
	})
	return
}
